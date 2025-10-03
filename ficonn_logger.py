#!/usr/bin/env python3
"""
CONSOLIDATED FICONN LOGGING SYSTEM
==================================
This replaces all the separate logging files with one comprehensive system.
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json
import argparse

class FiconnLogger:
    """
    Consolidated logging system for FICONN project.
    Replaces: terminal_logger.py, interactive_logger.py, log_command.py
    """
    
    def __init__(self, log_dir="logs", project_name="ficonn-project"):
        self.log_dir = Path(log_dir)
        self.project_name = project_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory structure
        self.log_dir.mkdir(exist_ok=True)
        self.session_log_dir = self.log_dir / f"{project_name}_{self.session_id}"
        self.session_log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Session metadata
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "project_name": project_name,
            "commands": [],
            "status": "active"
        }
        
        self.log_info(f"FICONN Logger initialized for session {self.session_id}")
        self.log_info(f"Log directory: {self.session_log_dir}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.session_log_dir / "ficonn_training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def log_error(self, message):
        """Log an error message"""
        self.logger.error(message)
    
    def log_warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def log_training_progress(self, epoch, train_acc, test_acc, loss, learning_rate=None):
        """Log training progress"""
        progress_msg = f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Loss: {loss:.4f}"
        if learning_rate:
            progress_msg += f", LR: {learning_rate:.6f}"
        self.log_info(progress_msg)
    
    def log_command(self, command, output=None, return_code=None, duration=None):
        """Log a command execution"""
        command_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "output": output,
            "return_code": return_code,
            "duration": duration
        }
        
        self.session_metadata["commands"].append(command_entry)
        
        # Log to file
        self.log_info(f"Command: {command}")
        if output:
            self.log_info(f"Output: {output}")
        if return_code is not None:
            self.log_info(f"Return code: {return_code}")
        if duration:
            self.log_info(f"Duration: {duration:.2f} seconds")
    
    def run_command_with_logging(self, command, shell=True, capture_output=True):
        """Run a command and log the results"""
        self.log_info(f"Executing: {command}")
        
        start_time = time.time()
        
        try:
            if capture_output:
                result = subprocess.run(
                    command, 
                    shell=shell, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8',
                    errors='replace'
                )
                stdout = result.stdout if result.stdout else ""
                stderr = result.stderr if result.stderr else ""
                output = stdout + stderr
                return_code = result.returncode
            else:
                result = subprocess.run(command, shell=shell)
                output = "Interactive command - output not captured"
                return_code = result.returncode
            
            duration = time.time() - start_time
            self.log_command(command, output, return_code, duration)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Command failed with exception: {str(e)}"
            self.log_error(error_msg)
            self.log_command(command, error_msg, -1, duration)
            raise
    
    def save_training_results(self, results_dict, filename="training_results.json"):
        """Save training results to file"""
        results_file = self.session_log_dir / filename
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        self.log_info(f"Training results saved to: {results_file}")
    
    def save_session_metadata(self):
        """Save session metadata to JSON file"""
        metadata_file = self.session_log_dir / "session_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)
    
    def end_session(self):
        """End the current session"""
        self.session_metadata["end_time"] = datetime.now().isoformat()
        self.session_metadata["status"] = "completed"
        self.save_session_metadata()
        self.log_info(f"Session {self.session_id} ended")
    
    def list_previous_sessions(self):
        """List all previous logging sessions"""
        sessions = []
        for session_dir in self.log_dir.glob(f"{self.project_name}_*"):
            if session_dir.is_dir():
                metadata_file = session_dir / "session_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        sessions.append(metadata)
                    except:
                        pass
        
        return sorted(sessions, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def get_session_summary(self, session_id=None):
        """Get summary of a specific session or current session"""
        if session_id is None:
            session_id = self.session_id
        
        sessions = self.list_previous_sessions()
        for session in sessions:
            if session['session_id'] == session_id:
                return session
        
        return None

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="FICONN Consolidated Logger")
    parser.add_argument("--list-sessions", action="store_true", help="List previous sessions")
    parser.add_argument("--session-id", help="Get summary of specific session")
    parser.add_argument("--command", help="Run a command with logging")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--project", default="ficonn-project", help="Project name")
    
    args = parser.parse_args()
    
    logger = FiconnLogger(args.log_dir, args.project)
    
    try:
        if args.list_sessions:
            sessions = logger.list_previous_sessions()
            print(f"\nPrevious Sessions ({len(sessions)}):")
            print("-" * 80)
            for session in sessions:
                print(f"Session ID: {session['session_id']}")
                print(f"Start Time: {session['start_time']}")
                print(f"Status: {session['status']}")
                print(f"Commands: {len(session.get('commands', []))}")
                print("-" * 40)
        
        elif args.session_id:
            summary = logger.get_session_summary(args.session_id)
            if summary:
                print(f"\nSession Summary: {args.session_id}")
                print(json.dumps(summary, indent=2))
            else:
                print(f"Session {args.session_id} not found")
        
        elif args.command:
            logger.run_command_with_logging(args.command)
        
        else:
            print("FICONN Logger initialized. Use --help for options.")
            print(f"Session ID: {logger.session_id}")
            print(f"Log directory: {logger.session_log_dir}")
    
    finally:
        logger.end_session()

if __name__ == "__main__":
    main()
