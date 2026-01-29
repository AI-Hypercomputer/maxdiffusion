import zmq
import sys
import os
import signal
import time
from typing import List
from maxdiffusion.inference.server.worker import DiffusionTPUWorker
from maxdiffusion.inference.server.schemas import DiffusionRequestState
from maxdiffusion.inference.server.protocol import send_pyobj, recv_pyobj
from maxdiffusion import max_logging

class DiffusionScheduler:
    """
    Main backend process for MaxDiffusion Serving.
    """
    def __init__(self, config_args: List[str], port: int = 5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.port}")
        
        max_logging.log(f"Scheduler initializing Worker with args: {config_args}")
        self.worker = DiffusionTPUWorker(config_args)
        self.running = True
        
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        max_logging.log("Scheduler received termination signal.")
        self.running = False

    def run(self):
        max_logging.log(f"Scheduler ready and listening on port {self.port}")
        
        while self.running:
            try:
                if self.socket.poll(timeout=1000):
                    identity, req_state = recv_pyobj(self.socket)
                    
                    if not isinstance(req_state, DiffusionRequestState):
                        max_logging.log(f"Received invalid request type: {type(req_state)}")
                        send_pyobj(self.socket, {"error": "Invalid request type"}, identity)
                        continue
                        
                    max_logging.log(f"Processing request {req_state.request_id}")
                    
                    try:
                        start_time = time.time()
                        # Process Request (Blocking call to TPU Worker)
                        responses = self.worker.process_request(req_state.request)
                        end_time = time.time()
                        
                        response = responses[0] # Assume single request logic for now
                        
                        result = {
                            "request_id": req_state.request_id,
                            "images": response.images, 
                            "latency": end_time - start_time
                        }
                        
                        send_pyobj(self.socket, result, identity)
                        
                    except Exception as e:
                        max_logging.log(f"Error processing request: {e}")
                        import traceback
                        traceback.print_exc()
                        send_pyobj(self.socket, {"error": str(e)}, identity)
                        
            except zmq.ZMQError as e:
                if self.running:
                    max_logging.log(f"ZMQ Error: {e}")
            except Exception as e:
                max_logging.log(f"Unexpected error: {e}")
                
        self.socket.close()
        self.context.term()
        max_logging.log("Scheduler shutdown complete.")

if __name__ == "__main__":
    # Args: script_name, config_path, [overrides...]
    # e.g. python -m maxdiffusion.inference.server.scheduler my_config.yaml
    if len(sys.argv) < 2:
        print("Usage: python -m maxdiffusion.inference.server.scheduler <config_path> [overrides...]")
        sys.exit(1)
        
    # pyconfig expects argv including script name
    config_args = sys.argv
    
    scheduler = DiffusionScheduler(config_args)
    scheduler.run()
