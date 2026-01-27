#!/usr/bin/env python3
"""
Heart Disease Risk Prediction System - Setup Script
Run this script to install dependencies and start the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def start_application():
    """Start the web application"""
    print("\n🚀 Starting Heart Disease Prediction System...")
    print("🌐 Application will be available at: http://localhost:5000")
    print("📝 Press Ctrl+C to stop the application\n")
    
    try:
        subprocess.check_call([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError:
        print("❌ Failed to start application")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🏥 Heart Disease Risk Prediction System - Setup")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install dependencies
    if install_requirements():
        # Start application
        start_application()
    else:
        print("❌ Setup failed. Please check your Python installation.")

if __name__ == "__main__":
    main()
