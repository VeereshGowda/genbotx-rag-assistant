"""
Document Processing Performance Test
Tests document ingestion and processing capabilities
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from src.vector_store_manager import VectorStoreManager
from src.content_manager import ContentManager

def test_document_processing_performance():
    """Test document processing capabilities"""
    print("📄 Testing Document Processing Performance...")
    
    # Get list of available documents
    documents_dir = Path("documents")
    available_docs = list(documents_dir.glob("*"))
    
    if not available_docs:
        print("❌ No documents found in documents directory")
        return {}
    
    print(f"📁 Found {len(available_docs)} documents:")
    for doc in available_docs:
        print(f"   • {doc.name} ({doc.stat().st_size / 1024:.1f} KB)")
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    content_manager = ContentManager()
    
    processing_results = {
        "total_documents": len(available_docs),
        "document_details": [],
        "total_processing_time": 0,
        "total_pages_processed": 0,
        "average_processing_speed": 0
    }
    
    total_start_time = time.time()
    
    for doc_path in available_docs:
        if doc_path.is_file() and doc_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
            print(f"\n   Processing: {doc_path.name}")
            
            start_time = time.time()
            try:
                # Process document
                if doc_path.suffix.lower() == '.pdf':
                    documents = doc_processor.process_pdf(str(doc_path))
                elif doc_path.suffix.lower() == '.docx':
                    documents = doc_processor.process_docx(str(doc_path))
                elif doc_path.suffix.lower() == '.txt':
                    documents = doc_processor.process_txt(str(doc_path))
                
                processing_time = time.time() - start_time
                
                doc_result = {
                    "filename": doc_path.name,
                    "file_size_kb": round(doc_path.stat().st_size / 1024, 1),
                    "format": doc_path.suffix.lower(),
                    "processing_time": round(processing_time, 3),
                    "chunks_created": len(documents) if documents else 0,
                    "processing_speed_kb_per_sec": round((doc_path.stat().st_size / 1024) / processing_time, 2),
                    "status": "success"
                }
                
                print(f"      ✅ Processed in {processing_time:.2f}s, {len(documents)} chunks created")
                
            except Exception as e:
                doc_result = {
                    "filename": doc_path.name,
                    "file_size_kb": round(doc_path.stat().st_size / 1024, 1),
                    "format": doc_path.suffix.lower(),
                    "processing_time": time.time() - start_time,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"      ❌ Failed: {e}")
            
            processing_results["document_details"].append(doc_result)
    
    total_processing_time = time.time() - total_start_time
    processing_results["total_processing_time"] = round(total_processing_time, 3)
    
    # Calculate averages
    successful_docs = [d for d in processing_results["document_details"] if d["status"] == "success"]
    if successful_docs:
        total_size = sum(d["file_size_kb"] for d in successful_docs)
        processing_results["total_size_processed_kb"] = total_size
        processing_results["average_processing_speed_kb_per_sec"] = round(total_size / total_processing_time, 2)
        processing_results["successful_documents"] = len(successful_docs)
        processing_results["failed_documents"] = len(processing_results["document_details"]) - len(successful_docs)
    
    print(f"\n✅ Document processing test completed")
    print(f"   Total documents: {len(processing_results['document_details'])}")
    print(f"   Successful: {processing_results.get('successful_documents', 0)}")
    print(f"   Total processing time: {total_processing_time:.2f}s")
    if successful_docs:
        print(f"   Average speed: {processing_results['average_processing_speed_kb_per_sec']:.2f} KB/s")
    
    return processing_results

def test_system_information():
    """Capture system information"""
    print("💻 Capturing System Information...")
    
    import platform
    import psutil
    import os
    
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_count": psutil.cpu_count(),
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "working_directory": os.getcwd()
    }
    
    print(f"   OS: {system_info['operating_system']} {system_info['os_version']}")
    print(f"   Python: {system_info['python_version']}")
    print(f"   CPU: {system_info['cpu_count']} cores, {system_info['cpu_usage_percent']}% usage")
    print(f"   Memory: {system_info['available_memory_gb']:.1f}/{system_info['total_memory_gb']:.1f} GB available")
    
    return system_info

if __name__ == "__main__":
    print("🧪 Document Processing & System Performance Test")
    print("="*60)
    
    # Test document processing
    doc_results = test_document_processing_performance()
    
    # Capture system info
    sys_info = test_system_information()
    
    # Combine results
    complete_results = {
        "document_processing": doc_results,
        "system_information": sys_info
    }
    
    # Save results
    output_file = "document_processing_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("🎉 Document processing test completed!")
