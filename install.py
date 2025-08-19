#!/usr/bin/env python3
"""
GenBotX Installation and Setup Script
Automated installation process for GenBotX RAG system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple
import yaml

class GenBotXInstaller:
    """Automated installer for GenBotX system"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        
    def check_requirements(self) -> List[str]:
        """Check system requirements"""
        issues = []
        
        # Check Python version
        if self.python_version < (3, 12):
            issues.append(f"Python 3.12+ required, found {self.python_version.major}.{self.python_version.minor}")
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                issues.append("Ollama not found in PATH")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Ollama not installed or not accessible")
        
        return issues
    
    def install_ollama_models(self) -> bool:
        """Install required Ollama models"""
        models = ['llama3.2', 'mxbai-embed-large']
        
        print("Installing Ollama models...")
        for model in models:
            print(f"  Installing {model}...")
            try:
                result = subprocess.run(['ollama', 'pull', model], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"  ✓ {model} installed successfully")
                else:
                    print(f"  ✗ Failed to install {model}: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout installing {model}")
                return False
            except Exception as e:
                print(f"  ✗ Error installing {model}: {e}")
                return False
        
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment configuration"""
        env_example = self.project_root / '.env.example'
        env_file = self.project_root / '.env'
        
        if env_example.exists() and not env_file.exists():
            print("Setting up environment configuration...")
            try:
                # Copy .env.example to .env
                with open(env_example, 'r') as src:
                    content = src.read()
                
                with open(env_file, 'w') as dst:
                    dst.write(content)
                
                print("  ✓ Environment file created (.env)")
                print("  📝 Edit .env file to customize settings if needed")
                return True
            except Exception as e:
                print(f"  ✗ Failed to create environment file: {e}")
                return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        print("Installing Python dependencies...")
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("  ✓ Dependencies installed successfully")
                return True
            else:
                print(f"  ✗ Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ✗ Error installing dependencies: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        directories = [
            'uploads',
            'documents',
            'vector_store',
            'logs',
            'test_uploads',
            'test_content_docs',
            'test_scraped_docs'
        ]
        
        print("Creating project directories...")
        for directory in directories:
            dir_path = self.project_root / directory
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"  ✓ {directory}/")
            except Exception as e:
                print(f"  ✗ Failed to create {directory}: {e}")
                return False
        
        return True
    
    def test_configuration(self) -> bool:
        """Test system configuration"""
        print("Testing configuration...")
        try:
            result = subprocess.run([sys.executable, 'test_config.py'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("  ✓ Configuration test passed")
                return True
            else:
                print(f"  ✗ Configuration test failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ✗ Error testing configuration: {e}")
            return False
    
    def initialize_system(self) -> bool:
        """Initialize the system"""
        print("Initializing GenBotX system...")
        try:
            result = subprocess.run([sys.executable, 'setup.py'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("  ✓ System initialized successfully")
                return True
            else:
                print(f"  ✗ System initialization failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ✗ Error initializing system: {e}")
            return False
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 GenBotX Installation Complete!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Start the web interface:")
        print("   streamlit run app.py")
        print("\n2. Or use the command line interface:")
        print("   python main.py")
        print("\n3. Access the web interface at:")
        print("   http://localhost:8501")
        print("\n4. Upload documents and start querying!")
        print("\nFor help and documentation:")
        print("- README.md for detailed instructions")
        print("- CONFIGURATION_SUMMARY.md for configuration options")
        print("- Check logs/ directory for troubleshooting")
        print("\n" + "="*60)
    
    def run_installation(self):
        """Run the complete installation process"""
        print("🚀 GenBotX Installation Script")
        print("=" * 40)
        
        # Check requirements
        print("\n1. Checking system requirements...")
        issues = self.check_requirements()
        if issues:
            print("❌ System requirements not met:")
            for issue in issues:
                print(f"   - {issue}")
            print("\nPlease resolve these issues and run the installer again.")
            return False
        print("  ✓ System requirements satisfied")
        
        # Create directories
        print("\n2. Creating project structure...")
        if not self.create_directories():
            return False
        
        # Setup environment
        print("\n3. Setting up environment...")
        if not self.setup_environment():
            return False
        
        # Install dependencies
        print("\n4. Installing dependencies...")
        if not self.install_dependencies():
            return False
        
        # Install Ollama models
        print("\n5. Installing AI models...")
        if not self.install_ollama_models():
            print("⚠️  Model installation failed. You can install them manually:")
            print("   ollama pull llama3.2")
            print("   ollama pull mxbai-embed-large")
        
        # Test configuration
        print("\n6. Testing configuration...")
        if not self.test_configuration():
            print("⚠️  Configuration test failed. Check the logs for details.")
        
        # Initialize system
        print("\n7. Initializing system...")
        if not self.initialize_system():
            print("⚠️  System initialization failed. You can run 'python setup.py' manually.")
        
        # Print success message and next steps
        self.print_next_steps()
        return True

if __name__ == "__main__":
    installer = GenBotXInstaller()
    success = installer.run_installation()
    sys.exit(0 if success else 1)
