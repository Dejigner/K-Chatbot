"""
Security Module for Klaro Academic Chatbot

This module provides security features including input validation, sanitization,
rate limiting, and system hardening to protect against common vulnerabilities
and ensure safe operation of the educational chatbot.
"""

import re
import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configuration for security settings"""
    max_query_length: int = 1000
    max_file_size_mb: int = 50
    allowed_file_extensions: List[str] = None
    rate_limit_requests_per_minute: int = 60
    rate_limit_window_minutes: int = 1
    enable_content_filtering: bool = True
    log_security_events: bool = True
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.pdf']

@dataclass
class SecurityEvent:
    """Structure for security events"""
    timestamp: float
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    source_ip: Optional[str]
    user_agent: Optional[str]
    details: str
    blocked: bool

class InputValidator:
    """
    Handles input validation and sanitization for user inputs.
    
    Features:
    - Query text validation and sanitization
    - File upload validation
    - Content filtering for inappropriate content
    - SQL injection and XSS prevention
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize the input validator.
        
        Args:
            config: Security configuration
        """
        self.config = config
        
        # Patterns for detecting malicious content
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"].*['\"])",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        self.inappropriate_content_patterns = [
            # Add patterns for inappropriate content detection
            # This is a simplified example - in production, use more sophisticated filtering
            r"\b(hate|violence|explicit)\b",
        ]
        
        # Compile patterns for efficiency
        self.compiled_sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self.compiled_content_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.inappropriate_content_patterns]
    
    def validate_query(self, query: str) -> Tuple[bool, str, str]:
        """
        Validate and sanitize a user query.
        
        Args:
            query: User input query
            
        Returns:
            Tuple of (is_valid, sanitized_query, error_message)
        """
        if not query or not isinstance(query, str):
            return False, "", "Query must be a non-empty string"
        
        # Check length
        if len(query) > self.config.max_query_length:
            return False, "", f"Query too long (max {self.config.max_query_length} characters)"
        
        # Check for SQL injection attempts
        for pattern in self.compiled_sql_patterns:
            if pattern.search(query):
                logger.warning(f"SQL injection attempt detected: {query[:100]}...")
                return False, "", "Invalid characters detected in query"
        
        # Check for XSS attempts
        for pattern in self.compiled_xss_patterns:
            if pattern.search(query):
                logger.warning(f"XSS attempt detected: {query[:100]}...")
                return False, "", "Invalid characters detected in query"
        
        # Check for inappropriate content
        if self.config.enable_content_filtering:
            for pattern in self.compiled_content_patterns:
                if pattern.search(query):
                    logger.warning(f"Inappropriate content detected: {query[:100]}...")
                    return False, "", "Content not appropriate for educational context"
        
        # Sanitize the query
        sanitized_query = self._sanitize_text(query)
        
        return True, sanitized_query, ""
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate an uploaded file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file extension
        if file_path.suffix.lower() not in self.config.allowed_file_extensions:
            return False, f"File type not allowed. Allowed types: {', '.join(self.config.allowed_file_extensions)}"
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return False, f"File too large (max {self.config.max_file_size_mb} MB)"
        
        # Basic file content validation for PDFs
        if file_path.suffix.lower() == '.pdf':
            if not self._validate_pdf_file(file_path):
                return False, "Invalid or corrupted PDF file"
        
        return True, ""
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by removing potentially harmful content.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Limit consecutive special characters
        text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?]{5,}', '...', text)
        
        return text.strip()
    
    def _validate_pdf_file(self, file_path: Path) -> bool:
        """
        Validate that a file is a proper PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    return False
            
            # Try to open with PyMuPDF for deeper validation
            import fitz
            doc = fitz.open(str(file_path))
            page_count = doc.page_count
            doc.close()
            
            return page_count > 0
            
        except Exception as e:
            logger.warning(f"PDF validation failed for {file_path}: {str(e)}")
            return False

class RateLimiter:
    """
    Implements rate limiting to prevent abuse and ensure fair usage.
    
    Features:
    - Per-IP rate limiting
    - Sliding window algorithm
    - Configurable limits and time windows
    - Automatic cleanup of old entries
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize the rate limiter.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_entries, daemon=True)
        self.cleanup_thread.start()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            client_id: Unique identifier for the client (e.g., IP address)
            
        Returns:
            Tuple of (is_allowed, requests_remaining)
        """
        current_time = time.time()
        window_start = current_time - (self.config.rate_limit_window_minutes * 60)
        
        with self.lock:
            # Remove old requests outside the window
            client_requests = self.requests[client_id]
            while client_requests and client_requests[0] < window_start:
                client_requests.popleft()
            
            # Check if limit is exceeded
            if len(client_requests) >= self.config.rate_limit_requests_per_minute:
                return False, 0
            
            # Add current request
            client_requests.append(current_time)
            
            # Calculate remaining requests
            remaining = self.config.rate_limit_requests_per_minute - len(client_requests)
            
            return True, remaining
    
    def _cleanup_old_entries(self):
        """Background thread to clean up old rate limit entries."""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                current_time = time.time()
                window_start = current_time - (self.config.rate_limit_window_minutes * 60 * 2)  # Keep extra buffer
                
                with self.lock:
                    clients_to_remove = []
                    
                    for client_id, requests in self.requests.items():
                        # Remove old requests
                        while requests and requests[0] < window_start:
                            requests.popleft()
                        
                        # Remove clients with no recent requests
                        if not requests:
                            clients_to_remove.append(client_id)
                    
                    for client_id in clients_to_remove:
                        del self.requests[client_id]
                        
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {str(e)}")

class SecurityMonitor:
    """
    Monitors security events and maintains security logs.
    
    Features:
    - Security event logging
    - Threat detection and alerting
    - Security metrics collection
    - Incident response coordination
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize the security monitor.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.events: List[SecurityEvent] = []
        self.lock = threading.Lock()
        
        # Security metrics
        self.metrics = {
            'total_events': 0,
            'blocked_requests': 0,
            'validation_failures': 0,
            'rate_limit_violations': 0,
            'file_validation_failures': 0
        }
    
    def log_event(self, 
                  event_type: str,
                  severity: str,
                  details: str,
                  blocked: bool = False,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None):
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            details: Event details
            blocked: Whether the request was blocked
            source_ip: Source IP address
            user_agent: User agent string
        """
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details,
            blocked=blocked
        )
        
        with self.lock:
            self.events.append(event)
            self.metrics['total_events'] += 1
            
            if blocked:
                self.metrics['blocked_requests'] += 1
            
            # Update specific metrics
            if event_type == 'validation_failure':
                self.metrics['validation_failures'] += 1
            elif event_type == 'rate_limit_violation':
                self.metrics['rate_limit_violations'] += 1
            elif event_type == 'file_validation_failure':
                self.metrics['file_validation_failures'] += 1
        
        # Log to system logger
        if self.config.log_security_events:
            log_message = f"Security Event [{severity.upper()}] {event_type}: {details}"
            if source_ip:
                log_message += f" (IP: {source_ip})"
            
            if severity in ['high', 'critical']:
                logger.warning(log_message)
            else:
                logger.info(log_message)
    
    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """
        Get recent security events.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent security events
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            return [event for event in self.events if event.timestamp >= cutoff_time]
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get security metrics.
        
        Returns:
            Dictionary with security metrics
        """
        with self.lock:
            recent_events = self.get_recent_events(24)
            
            return {
                **self.metrics,
                'recent_events_24h': len(recent_events),
                'high_severity_events_24h': len([e for e in recent_events if e.severity in ['high', 'critical']]),
                'blocked_requests_24h': len([e for e in recent_events if e.blocked]),
                'unique_ips_24h': len(set(e.source_ip for e in recent_events if e.source_ip))
            }

class SecurityManager:
    """
    Main security manager that coordinates all security components.
    
    Features:
    - Centralized security policy enforcement
    - Integration of all security components
    - Security configuration management
    - Threat response coordination
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize the security manager.
        
        Args:
            config: Security configuration (uses defaults if None)
        """
        self.config = config or SecurityConfig()
        
        # Initialize security components
        self.validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.monitor = SecurityMonitor(self.config)
        
        logger.info("Security manager initialized")
    
    def validate_request(self, 
                        query: str,
                        client_id: str,
                        source_ip: Optional[str] = None,
                        user_agent: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Validate a complete request including rate limiting and input validation.
        
        Args:
            query: User query to validate
            client_id: Client identifier for rate limiting
            source_ip: Source IP address
            user_agent: User agent string
            
        Returns:
            Tuple of (is_valid, sanitized_query, error_message)
        """
        # Check rate limits first
        is_allowed, remaining = self.rate_limiter.is_allowed(client_id)
        if not is_allowed:
            self.monitor.log_event(
                event_type='rate_limit_violation',
                severity='medium',
                details=f'Rate limit exceeded for client {client_id}',
                blocked=True,
                source_ip=source_ip,
                user_agent=user_agent
            )
            return False, "", "Rate limit exceeded. Please try again later."
        
        # Validate input
        is_valid, sanitized_query, error_message = self.validator.validate_query(query)
        if not is_valid:
            self.monitor.log_event(
                event_type='validation_failure',
                severity='medium',
                details=f'Input validation failed: {error_message}',
                blocked=True,
                source_ip=source_ip,
                user_agent=user_agent
            )
            return False, "", error_message
        
        # Log successful validation
        self.monitor.log_event(
            event_type='request_validated',
            severity='low',
            details=f'Request validated successfully for client {client_id}',
            blocked=False,
            source_ip=source_ip,
            user_agent=user_agent
        )
        
        return True, sanitized_query, ""
    
    def validate_file_upload(self, 
                           file_path: Path,
                           client_id: str,
                           source_ip: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate a file upload.
        
        Args:
            file_path: Path to uploaded file
            client_id: Client identifier
            source_ip: Source IP address
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check rate limits
        is_allowed, remaining = self.rate_limiter.is_allowed(client_id)
        if not is_allowed:
            self.monitor.log_event(
                event_type='rate_limit_violation',
                severity='medium',
                details=f'File upload rate limit exceeded for client {client_id}',
                blocked=True,
                source_ip=source_ip
            )
            return False, "Rate limit exceeded. Please try again later."
        
        # Validate file
        is_valid, error_message = self.validator.validate_file(file_path)
        if not is_valid:
            self.monitor.log_event(
                event_type='file_validation_failure',
                severity='medium',
                details=f'File validation failed: {error_message}',
                blocked=True,
                source_ip=source_ip
            )
            return False, error_message
        
        # Log successful validation
        self.monitor.log_event(
            event_type='file_validated',
            severity='low',
            details=f'File {file_path.name} validated successfully',
            blocked=False,
            source_ip=source_ip
        )
        
        return True, ""
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status.
        
        Returns:
            Dictionary with security status information
        """
        return {
            'config': {
                'max_query_length': self.config.max_query_length,
                'max_file_size_mb': self.config.max_file_size_mb,
                'rate_limit_per_minute': self.config.rate_limit_requests_per_minute,
                'content_filtering_enabled': self.config.enable_content_filtering
            },
            'metrics': self.monitor.get_security_metrics(),
            'recent_events': len(self.monitor.get_recent_events(1)),  # Last hour
            'status': 'active'
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize security manager
    config = SecurityConfig(
        max_query_length=500,
        rate_limit_requests_per_minute=10,
        enable_content_filtering=True
    )
    
    security_manager = SecurityManager(config)
    
    # Test query validation
    test_queries = [
        "What is photosynthesis?",  # Valid
        "SELECT * FROM users WHERE id=1",  # SQL injection attempt
        "<script>alert('xss')</script>What is biology?",  # XSS attempt
        "A" * 1000,  # Too long
    ]
    
    print("Testing query validation:")
    for i, query in enumerate(test_queries, 1):
        is_valid, sanitized, error = security_manager.validate_request(
            query, f"client_{i}", "192.168.1.1", "TestAgent/1.0"
        )
        print(f"{i}. Valid: {is_valid}, Error: {error}")
        if is_valid:
            print(f"   Sanitized: {sanitized[:50]}...")
    
    # Test rate limiting
    print("\nTesting rate limiting:")
    for i in range(12):  # Exceed rate limit
        is_valid, _, error = security_manager.validate_request(
            "Test query", "rate_test_client", "192.168.1.2"
        )
        if not is_valid:
            print(f"Request {i+1}: Blocked - {error}")
            break
        else:
            print(f"Request {i+1}: Allowed")
    
    # Print security status
    print("\nSecurity Status:")
    status = security_manager.get_security_status()
    for key, value in status.items():
        print(f"{key}: {value}")

