# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, Response
import os
import sys
from datetime import datetime
import json
import threading
from queue import Queue
import uuid
import time
import socket
from dotenv import load_dotenv
import pandas as pd
import io

# Load environment variables from .env file
load_dotenv()

# Ensure required environment variables are provided via .env or hosting platform secrets.

# Add the synthesis engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'synthesis_engine'))

from synthesis_engine.analysis import SynthesisAnalyzer
from synthesis_engine.utils import initialize_session, get_session_data, update_session_data
from synthesis_engine.api_buyer_finder import ApiBuyerFinder  # Import the new API Buyer Finder
from synthesis_engine.api_buyer_discovery import ApiBuyerDiscoveryService
from synthesis_engine.api_manufacturer_service import ApiManufacturerService
from synthesis_engine.api_manufacturer_discovery import ApiManufacturerDiscoveryService

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')  # Use environment variable for security

# Global analyzer instance - lazy loaded to reduce memory usage
analyzer = None
api_buyer_finder = None # Global instance for API Buyer Finder
api_buyer_discovery = None
api_manufacturer_service = None
api_manufacturer_discovery = None
new_manufacturer_service = None
new_manufacturer_discovery = None

# Dictionary to store threading.Event objects for stopping analysis processes
stop_events = {}
# Dictionary to store queues for sending progress updates via SSE
progress_queues = {}

# Lazy initialization function to reduce startup memory usage
def initialize_services():
    """Lazy initialization of heavy services to reduce memory footprint"""
    global analyzer, api_buyer_finder, api_buyer_discovery
    global api_manufacturer_service, api_manufacturer_discovery
    global new_manufacturer_service, new_manufacturer_discovery
    
    if analyzer is None:
        with app.app_context():
            analyzer = SynthesisAnalyzer()
    
    if api_buyer_finder is None:
        with app.app_context():
            api_buyer_finder = ApiBuyerFinder()
            # Use ApiBuyerDiscoveryService for verified discovery
            api_buyer_discovery = ApiBuyerDiscoveryService()
    elif api_buyer_discovery is None:
        with app.app_context():
            # Use ApiBuyerDiscoveryService for verified discovery
            api_buyer_discovery = ApiBuyerDiscoveryService()
    
    if api_manufacturer_service is None:
        with app.app_context():
            api_manufacturer_service = ApiManufacturerService()
            api_manufacturer_discovery = ApiManufacturerDiscoveryService(api_manufacturer_service)
            
            new_db_path = os.environ.get(
                "NEW_SQLITE_DB_FILENAME",
                os.path.join(os.path.dirname(__file__), "new_manufacturers.db")
            )
            new_manufacturer_service = ApiManufacturerService(db_filename=new_db_path)
            new_manufacturer_discovery = ApiManufacturerDiscoveryService(new_manufacturer_service)

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_synthesis():
    """Main synthesis analysis endpoint"""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        
        # Extract parameters
        api_name = data.get('api_name', '').strip()
        supplier_pref = data.get('supplier_preference', '').strip()
        search_depth = data.get('search_depth', 'deep')
        include_alternatives = data.get('include_alternatives', True)
        focus_high_yield = data.get('focus_high_yield', True)
        viability_threshold = data.get('viability_threshold', 75)
        
        if not api_name:
            return jsonify({'error': 'API name is required'}), 400
        
        # Initialize session for this analysis
        session_id = initialize_session(api_name)
        
        # Create a stop event and a progress queue for this session
        stop_events[session_id] = threading.Event()
        progress_queues[session_id] = Queue()

        # Define a progress callback function
        def progress_callback(percentage, message):
            if session_id in progress_queues:
                progress_queues[session_id].put({'percentage': percentage, 'message': message})

        # Run the analysis in a separate thread to allow progress updates and stopping
        # The actual result will be stored in the session data when complete
        def run_analysis_in_thread():
            try:
                result = analyzer.run_full_analysis(
                    api_name=api_name,
                    supplier_preference=supplier_pref,
                    search_depth=search_depth,
                    include_alternatives=include_alternatives,
                    focus_high_yield=focus_high_yield,
                    viability_threshold=viability_threshold,
                    progress_callback=progress_callback,
                    stop_event=stop_events[session_id]
                )
                update_session_data(session_id, {
                    'analysis_complete': True,
                    'results': result,
                    'timestamp': datetime.now().isoformat()
                })
                if session_id in progress_queues:
                    progress_queues[session_id].put({'percentage': 100, 'message': 'Analysis complete!'})
            except Exception as e:
                print(f"Error in analysis thread for session {session_id}: {e}")
                update_session_data(session_id, {
                    'analysis_complete': True,
                    'results': {'success': False, 'error': f'Analysis failed: {str(e)}'},
                    'timestamp': datetime.now().isoformat()
                })
                if session_id in progress_queues:
                    progress_queues[session_id].put({'percentage': 0, 'message': f'Analysis failed: {str(e)}', 'error': True})
            finally:
                # Clean up the queue and event after analysis (or stop)
                if session_id in progress_queues:
                    progress_queues[session_id].put(None) # Sentinel to signal end of stream
                if session_id in stop_events: # Ensure cleanup if analysis was stopped manually
                    del stop_events[session_id]
                if session_id in progress_queues:
                    del progress_queues[session_id]

        analysis_thread = threading.Thread(target=run_analysis_in_thread)
        analysis_thread.start()
        
        # Return session_id immediately, frontend will poll for progress
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Analysis started in background.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    def generate():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'percentage': 0, 'message': 'Invalid session or analysis not started.'})}\n\n"
            return

        queue = progress_queues[session_id]
        while True:
            item = queue.get()
            if item is None: # Sentinel value for end of stream
                break
            yield f"data: {json.dumps(item)}\n\n"

        # Clean up queue after stream ends
        if session_id in progress_queues: # Check again, might have been cleaned by analysis thread
            del progress_queues[session_id]

    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis_endpoint():
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        if session_id in stop_events:
            stop_events[session_id].set()  # Signal the thread to stop
            return jsonify({'success': True, 'message': f'Stop signal sent for session {session_id}'})
        else:
            return jsonify({'success': False, 'message': 'No active analysis for this session'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint for follow-up questions"""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        
        session_id = data.get('session_id')
        message = data.get('message', '').strip()
        
        if not session_id or not message:
            return jsonify({'error': 'Session ID and message are required'}), 400
        
        # Get session data
        session_data = get_session_data(session_id)
        if not session_data:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Generate chatbot response
        response = analyzer.chat_response(message, session_data)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>')
def get_session(session_id):
    """Get session data"""
    try:
        session_data = get_session_data(session_id)
        if not session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        # Add this debug print to inspect the session data before sending to frontend
        print(f"[DEBUG] Session data for {session_id}: {json.dumps(session_data, indent=2)}")
        
        return jsonify({
            'success': True,
            'data': session_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_route', methods=['POST'])
def predict_route():
    initialize_services()  # Lazy load services when needed
    data = request.json
    api_name = data.get('api_name')
    country_preference = data.get('country_preference', '')
    criteria = data.get('criteria', '')
    existing_session_id = data.get('session_id') # Get session_id from frontend if it exists

    if not api_name:
        return jsonify({'error': 'API name is required.'}), 400

    try:
        session_id = existing_session_id if existing_session_id else str(uuid.uuid4())

        # Initialize session data if it's a new session or if it's missing for an existing ID
        if session_id not in progress_queues: # Use progress_queues to check if session exists
            initialize_session(api_name, session_id) # Ensure initialize_session accepts session_id
        
        stop_event = threading.Event()
        progress_queue = Queue()
        stop_events[session_id] = stop_event
        progress_queues[session_id] = progress_queue

        def prediction_task():
            try:
                def progress_callback(progress, message):
                    if session_id in progress_queues:
                        progress_queues[session_id].put({'status': 'progress', 'progress': progress, 'message': message})
                
                result = analyzer.predict_synthesis_route(
                    api_name, 
                    country_preference, 
                    criteria, 
                    progress_callback,
                    stop_event
                )
                print(f"[DEBUG] AI prediction raw result for session {session_id}:\n{result}") # Log the raw result
                # Store the predicted route in the session data
                update_session_data(session_id, {
                    'ai_predicted_route': result.get('result', 'No predicted route found.'),
                    'prediction_complete': True
                })
                if session_id in progress_queues:
                    progress_queues[session_id].put({'status': 'complete', 'result': result})
            except Exception as e:
                print(f"Prediction error for session {session_id}: {e}")
                if session_id in progress_queues:
                    progress_queues[session_id].put({'status': 'error', 'message': str(e)})
            finally:
                # Clean up the queue and event after prediction (or stop)
                if session_id in progress_queues:
                    progress_queues[session_id].put(None) # Sentinel to signal end of stream
                if session_id in stop_events: # Ensure cleanup if prediction was stopped manually
                    del stop_events[session_id]
                if session_id in progress_queues:
                    del progress_queues[session_id]

        thread = threading.Thread(target=prediction_task)
        thread.start()

        return jsonify({'session_id': session_id})

    except Exception as e:
        print(f"[DEBUG] Error in predict_route endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction_progress/<session_id>')
def prediction_progress(session_id):
    def generate():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Invalid session ID'})}\n\n"
            return

        queue = progress_queues[session_id]
        while True:
            try:
                item = queue.get()
                if item is None: # Sentinel value for end of stream
                    print(f"[DEBUG] SSE for session {session_id}: Received sentinel, breaking stream.")
                    break
                
                print(f"[DEBUG] SSE for session {session_id}: Raw item from queue: {item}")
                json_data = json.dumps(item)
                print(f"[DEBUG] SSE for session {session_id}: Yielding JSON: {json_data}")
                yield f"data: {json_data}\n\n"
                
                if item.get('status') in ['complete', 'error']:
                    print(f"[DEBUG] SSE for session {session_id}: Status is {item.get('status')}, breaking stream.")
                    break
            except Exception as e:
                print(f"Error generating prediction progress for session {session_id}: {e}")
                yield f"data: {json.dumps({'status': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                break

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/visualize_reaction', methods=['POST'])
def visualize_reaction():
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        reaction_smiles = data.get('reaction_smiles')
        if not reaction_smiles:
            return jsonify({'error': 'Reaction SMILES is required.'}), 400

        base64_image = analyzer._generate_reaction_image(reaction_smiles)

        if base64_image:
            return jsonify({'success': True, 'image': base64_image}), 200
        else:
            return jsonify({'success': False, 'error': 'Failed to generate reaction image.'}), 500

    except Exception as e:
        print(f"[DEBUG] Error in visualize_reaction endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/stop_prediction', methods=['POST'])
def stop_prediction():
    data = request.json
    session_id = data.get('session_id') 

    if session_id and session_id in stop_events:
        stop_events[session_id].set()
        return jsonify({'message': f'Prediction for session {session_id} stopping.'}), 200
    return jsonify({'error': 'No active prediction found for this session.'}), 404

@app.route('/api/find_buyers', methods=['POST'])
def find_buyers():
    """Endpoint to find potential API buyers for a given API name."""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        api_name = data.get('api_name')
        country = data.get('country')  # Extract country from request
        verified = data.get('verified', True)

        if not api_name or not country:  # Both are now required
            return jsonify({'error': 'API name and country are required'}), 400

        # VERIFIED MODE: use only evidence-driven V2 discovery (no legacy fallback)
        if verified:
            if not api_buyer_discovery:
                return jsonify({'success': False, 'error': 'Verified buyer discovery service not available.'}), 500

            discovery_result = api_buyer_discovery.discover(api_name, country)
            # Always return the discovery_result as-is (success may be True or False)
            return jsonify(discovery_result)

        # UNVERIFIED MODE: use legacy finder (may return heuristic / potential leads)
        results = api_buyer_finder.find_api_buyers(api_name, country)  # Pass both api_name and country
        return jsonify(results)  # Return the full results dictionary directly
    except Exception as e:
        print(f"[DEBUG] Error in find_buyers endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/find_manufacturers', methods=['POST'])
def find_manufacturers():
    """Search API manufacturers stored in the legacy Excel-backed SQLite database."""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        api_name = data.get('api_name', '').strip()
        country = data.get('country', '').strip()

        if not api_name or not country:
            return jsonify({'success': False, 'error': 'API name and country are required'}), 400

        records = api_manufacturer_service.query(api_name, country)
        return jsonify({
            'success': True,
            'records': records,
            'requested_api': api_name,
            'requested_country': country
        })
    except Exception as e:
        print(f"[DEBUG] Error in find_manufacturers endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/api/discover_manufacturers', methods=['POST'])
def discover_manufacturers():
    """Run Google discovery and persist new manufacturers into the discovery database."""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        api_name = data.get('api_name', '').strip()
        country = data.get('country', '').strip()

        if not api_name or not country:
            return jsonify({'success': False, 'error': 'API name and country are required'}), 400

        discovery_result = api_manufacturer_discovery.discover(api_name, country)
        if not discovery_result.get('success', False):
            return jsonify({'success': False, 'error': discovery_result.get('error', 'Discovery failed')}), 500

        return jsonify({
            'success': True,
            'existing_records': discovery_result.get('existing_records', []),
            'new_records': discovery_result.get('new_records', []),
            'all_records': discovery_result.get('all_records', []),
            'inserted_count': discovery_result.get('inserted_count', 0),
            'requested_api': api_name,
            'requested_country': country
        })
    except Exception as e:
        print(f"[DEBUG] Error in find_manufacturers endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/download_buyers', methods=['POST'])
def download_buyers():
    """Download API buyers data as CSV or Excel - includes ALL entries from database"""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        api_name = data.get('api_name', '').strip()
        country = data.get('country', '').strip()
        format_type = data.get('format', 'csv').lower()  # 'csv' or 'excel'
        
        if not api_name or not country:
            return jsonify({'success': False, 'error': 'API name and country are required'}), 400
        
        # Fetch ALL data directly from database (existing + newly added)
        engine = api_buyer_finder.get_db_engine()
        if not engine:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        from sqlalchemy import text
        query = text("""
            SELECT company, form, strength, additional_info, url, verification_source, 
                   confidence, api, country, created_at, updated_at
            FROM viruj 
            WHERE api = :api AND country = :country
            ORDER BY created_at DESC
        """)
        
        with engine.begin() as conn:
            result = conn.execute(query, {"api": api_name, "country": country})
            rows = result.fetchall()
        
        if not rows:
            return jsonify({'success': False, 'error': 'No data found in database'}), 400
        
        # Create DataFrame with all columns
        df = pd.DataFrame(rows, columns=[
            'Company', 'Form', 'Strength', 'Additional Info', 'URL', 
            'Verification Source', 'Confidence', 'API', 'Country', 
            'Created At', 'Updated At'
        ])
        
        # Prepare filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_api = "".join(c for c in api_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_country = "".join(c for c in country if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"API_Buyers_{safe_api}_{safe_country}_{timestamp}"
        
        if format_type == 'excel':
            # Create Excel file in memory
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='API Buyers')
                output.seek(0)
                excel_data = output.getvalue()
                
                if len(excel_data) == 0:
                    return jsonify({'success': False, 'error': 'Generated Excel file is empty'}), 500
                
                return Response(
                    excel_data,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}.xlsx"',
                        'Content-Length': str(len(excel_data))
                    }
                )
            except Exception as e:
                print(f"[DEBUG] Excel generation error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'Excel generation failed: {str(e)}'}), 500
        else:
            # Create CSV file in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}.csv"'
                }
            )
            
    except Exception as e:
        print(f"[DEBUG] Error in download_buyers endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/download_manufacturers', methods=['POST'])
def download_manufacturers():
    """Download API manufacturers data as CSV or Excel - includes ALL entries from database"""
    try:
        initialize_services()  # Lazy load services when needed
        data = request.get_json()
        api_name = data.get('api_name', '').strip()
        country = data.get('country', '').strip()
        format_type = data.get('format', 'csv').lower()  # 'csv' or 'excel'
        
        if not api_name or not country:
            return jsonify({'success': False, 'error': 'API name and country are required'}), 400
        
        # Get ALL records directly from database (includes existing + newly discovered)
        records = api_manufacturer_service.query(api_name, country)
        
        if not records:
            return jsonify({'success': False, 'error': 'No data found in database'}), 400
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Reorder columns for better readability
        column_order = ['api_name', 'manufacturer', 'country', 'usdmf', 'cep', 'source_name', 'source_url', 'source_file', 'imported_at']
        df = df[[col for col in column_order if col in df.columns]]
        
        # Prepare filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_api = "".join(c for c in api_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_country = "".join(c for c in country if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"API_Manufacturers_{safe_api}_{safe_country}_{timestamp}"
        
        if format_type == 'excel':
            # Create Excel file in memory
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='API Manufacturers')
                output.seek(0)
                excel_data = output.getvalue()
                
                if len(excel_data) == 0:
                    return jsonify({'success': False, 'error': 'Generated Excel file is empty'}), 500
                
                return Response(
                    excel_data,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}.xlsx"',
                        'Content-Length': str(len(excel_data))
                    }
                )
            except Exception as e:
                print(f"[DEBUG] Excel generation error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'Excel generation failed: {str(e)}'}), 500
        else:
            # Create CSV file in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}.csv"'
                }
            )
            
    except Exception as e:
        print(f"[DEBUG] Error in download_manufacturers endpoint: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500



if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('synthesis_engine', exist_ok=True)
    
    # Initialize analyzer
    with app.app_context():
        analyzer = SynthesisAnalyzer()
    
    # Determine LAN IP for convenience logging
    def get_local_ip():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except Exception:
            return '127.0.0.1'

    # Use PORT from environment (for cloud platforms) or default to 5000
    # Replit uses port 8080, Render uses PORT env var
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    local_ip = get_local_ip()
    print(f"\nApp is starting...\n")
    print(f"Local access:   http://127.0.0.1:{port}")
    print(f"LAN access:     http://{local_ip}:{port}")
    print("Note: Ensure your firewall allows inbound connections on this port.")

    # Run the Flask app (bound to all interfaces for LAN access)
    app.run(debug=False, host=host, port=port)