"""
Comprehensive Performance Test for GenBotX
This script runs extensive tests and captures real performance metrics for publication.
"""

import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_engine import RAGEngine
from src.vector_store_manager import VectorStoreManager
from src.document_processor import DocumentProcessor
from src.content_manager import ContentManager
from src.config import get_config

class PerformanceTestSuite:
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "system_config": {},
            "initialization_metrics": {},
            "query_performance": {},
            "confidence_analysis": {},
            "document_processing": {},
            "vector_store_metrics": {},
            "memory_usage": {},
            "error_analysis": {}
        }
        
        self.test_queries = [
            "What is the Kuru kingdom?",
            "Describe the Anglo-Mysore Wars",
            "Tell me about Krishnadevaraya",
            "What were the main events in the Transformers movie series?",
            "Explain the significance of Mysore in Indian history",
            "What are the key characteristics of the Kuru dynasty?",
            "How did the Anglo-Mysore Wars impact South India?",
            "What was Krishnadevaraya's contribution to literature?",
            "Describe the plot of the first Transformers movie"
        ]
        
    def capture_system_config(self):
        """Capture current system configuration"""
        print("📋 Capturing System Configuration...")
        config = get_config()
        
        self.results["system_config"] = {
            "llm_model": config.get("llm.model"),
            "llm_temperature": config.get("llm.temperature"),
            "llm_max_tokens": config.get("llm.max_tokens"),
            "embedding_model": config.get("embeddings.model"),
            "similarity_search_k": config.get("vector_store.similarity_search_k"),
            "similarity_threshold": config.get("vector_store.similarity_threshold"),
            "chunk_size": config.get("text_splitter.chunk_size"),
            "chunk_overlap": config.get("text_splitter.chunk_overlap"),
            "confidence_threshold": config.get("rag_pipeline.confidence_threshold"),
            "enable_reasoning": config.get("rag_pipeline.enable_reasoning"),
            "enable_quality_check": config.get("rag_pipeline.enable_quality_check")
        }
        
        print("✅ System configuration captured")
        
    def test_initialization_performance(self):
        """Test system initialization time"""
        print("🚀 Testing System Initialization...")
        
        start_time = time.time()
        try:
            rag_engine = RAGEngine()
            init_time = time.time() - start_time
            
            self.results["initialization_metrics"] = {
                "initialization_time": round(init_time, 3),
                "status": "success",
                "components_loaded": ["RAGEngine", "VectorStore", "DocumentProcessor", "ContentManager"]
            }
            
            print(f"✅ System initialized in {init_time:.3f} seconds")
            return rag_engine
            
        except Exception as e:
            self.results["initialization_metrics"] = {
                "initialization_time": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Initialization failed: {e}")
            return None
    
    def test_query_performance(self, rag_engine):
        """Test query response performance"""
        if not rag_engine:
            print("❌ Skipping query tests - RAG engine not available")
            return
            
        print("🔍 Testing Query Performance...")
        
        query_results = []
        response_times = []
        confidence_scores = []
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"   Query {i}/{len(self.test_queries)}: {query[:50]}...")
            
            start_time = time.time()
            try:
                result = rag_engine.query(query)
                response_time = time.time() - start_time
                
                response_times.append(response_time)
                confidence_scores.append(result.get("confidence_score", 0.0))
                
                query_result = {
                    "query": query,
                    "response_time": round(response_time, 3),
                    "confidence_score": round(result.get("confidence_score", 0.0), 3),
                    "response_length": len(result.get("response", "")),
                    "documents_retrieved": len(result.get("retrieved_docs", [])),
                    "reasoning_steps": len(result.get("reasoning_steps", [])),
                    "status": "success"
                }
                
                print(f"      ✅ Response in {response_time:.2f}s, Confidence: {result.get('confidence_score', 0):.1%}")
                
            except Exception as e:
                query_result = {
                    "query": query,
                    "response_time": time.time() - start_time,
                    "confidence_score": 0.0,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"      ❌ Failed: {e}")
            
            query_results.append(query_result)
            time.sleep(0.5)  # Brief pause between queries
        
        # Calculate statistics
        if response_times:
            self.results["query_performance"] = {
                "total_queries": len(self.test_queries),
                "successful_queries": len([r for r in query_results if r["status"] == "success"]),
                "failed_queries": len([r for r in query_results if r["status"] == "failed"]),
                "average_response_time": round(statistics.mean(response_times), 3),
                "median_response_time": round(statistics.median(response_times), 3),
                "min_response_time": round(min(response_times), 3),
                "max_response_time": round(max(response_times), 3),
                "std_response_time": round(statistics.stdev(response_times) if len(response_times) > 1 else 0, 3),
                "success_rate": round(len([r for r in query_results if r["status"] == "success"]) / len(self.test_queries), 3),
                "detailed_results": query_results
            }
        
        print(f"✅ Query performance test completed")
    
    def analyze_confidence_distribution(self):
        """Analyze confidence score distribution"""
        print("📊 Analyzing Confidence Distribution...")
        
        if "query_performance" not in self.results or not self.results["query_performance"].get("detailed_results"):
            print("❌ No query results available for confidence analysis")
            return
        
        confidence_scores = [
            r["confidence_score"] for r in self.results["query_performance"]["detailed_results"] 
            if r["status"] == "success"
        ]
        
        if not confidence_scores:
            print("❌ No successful queries for confidence analysis")
            return
        
        # Categorize confidence scores
        high_confidence = [s for s in confidence_scores if s >= 0.8]
        medium_confidence = [s for s in confidence_scores if 0.6 <= s < 0.8]
        low_confidence = [s for s in confidence_scores if s < 0.6]
        
        self.results["confidence_analysis"] = {
            "total_responses": len(confidence_scores),
            "average_confidence": round(statistics.mean(confidence_scores), 3),
            "median_confidence": round(statistics.median(confidence_scores), 3),
            "min_confidence": round(min(confidence_scores), 3),
            "max_confidence": round(max(confidence_scores), 3),
            "std_confidence": round(statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0, 3),
            "high_confidence_count": len(high_confidence),
            "medium_confidence_count": len(medium_confidence),
            "low_confidence_count": len(low_confidence),
            "high_confidence_percentage": round(len(high_confidence) / len(confidence_scores) * 100, 1),
            "medium_confidence_percentage": round(len(medium_confidence) / len(confidence_scores) * 100, 1),
            "low_confidence_percentage": round(len(low_confidence) / len(confidence_scores) * 100, 1),
            "confidence_scores": confidence_scores
        }
        
        print(f"✅ Confidence analysis completed")
        print(f"   High Confidence (≥80%): {len(high_confidence)}/{len(confidence_scores)} ({len(high_confidence) / len(confidence_scores) * 100:.1f}%)")
        print(f"   Medium Confidence (60-80%): {len(medium_confidence)}/{len(confidence_scores)} ({len(medium_confidence) / len(confidence_scores) * 100:.1f}%)")
        print(f"   Low Confidence (<60%): {len(low_confidence)}/{len(confidence_scores)} ({len(low_confidence) / len(confidence_scores) * 100:.1f}%)")
    
    def test_vector_store_performance(self):
        """Test vector store operations"""
        print("🗄️ Testing Vector Store Performance...")
        
        try:
            vector_store = VectorStoreManager()
            
            # Test similarity search performance
            test_query = "What is the Kuru kingdom?"
            
            start_time = time.time()
            results = vector_store.similarity_search(test_query, k=10)
            search_time = time.time() - start_time
            
            start_time = time.time()
            results_with_scores = vector_store.similarity_search_with_score(test_query, k=10)
            search_with_scores_time = time.time() - start_time
            
            self.results["vector_store_metrics"] = {
                "search_time": round(search_time, 3),
                "search_with_scores_time": round(search_with_scores_time, 3),
                "documents_found": len(results),
                "documents_with_scores_found": len(results_with_scores),
                "status": "success"
            }
            
            if results_with_scores:
                scores = [score for _, score in results_with_scores]
                self.results["vector_store_metrics"].update({
                    "best_similarity_score": round(min(scores), 3),
                    "worst_similarity_score": round(max(scores), 3),
                    "average_similarity_score": round(statistics.mean(scores), 3)
                })
            
            print(f"✅ Vector store test completed")
            print(f"   Search time: {search_time:.3f}s")
            print(f"   Documents found: {len(results)}")
            
        except Exception as e:
            self.results["vector_store_metrics"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Vector store test failed: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 GENBOTX PERFORMANCE TEST REPORT")
        print("="*60)
        
        # System Configuration
        print("\n🔧 SYSTEM CONFIGURATION:")
        config = self.results["system_config"]
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Initialization
        print("\n🚀 INITIALIZATION METRICS:")
        init = self.results["initialization_metrics"]
        print(f"   Status: {init.get('status', 'unknown')}")
        print(f"   Time: {init.get('initialization_time', 0):.3f} seconds")
        
        # Query Performance
        print("\n🔍 QUERY PERFORMANCE:")
        if "query_performance" in self.results:
            perf = self.results["query_performance"]
            print(f"   Total Queries: {perf.get('total_queries', 0)}")
            print(f"   Success Rate: {perf.get('success_rate', 0)*100:.1f}%")
            print(f"   Average Response Time: {perf.get('average_response_time', 0):.3f}s")
            print(f"   Median Response Time: {perf.get('median_response_time', 0):.3f}s")
            print(f"   Min Response Time: {perf.get('min_response_time', 0):.3f}s")
            print(f"   Max Response Time: {perf.get('max_response_time', 0):.3f}s")
        
        # Confidence Analysis
        print("\n📊 CONFIDENCE ANALYSIS:")
        if "confidence_analysis" in self.results:
            conf = self.results["confidence_analysis"]
            print(f"   Average Confidence: {conf.get('average_confidence', 0)*100:.1f}%")
            print(f"   High Confidence (≥80%): {conf.get('high_confidence_percentage', 0):.1f}%")
            print(f"   Medium Confidence (60-80%): {conf.get('medium_confidence_percentage', 0):.1f}%")
            print(f"   Low Confidence (<60%): {conf.get('low_confidence_percentage', 0):.1f}%")
        
        # Vector Store
        print("\n🗄️ VECTOR STORE METRICS:")
        if "vector_store_metrics" in self.results:
            vs = self.results["vector_store_metrics"]
            print(f"   Search Time: {vs.get('search_time', 0):.3f}s")
            print(f"   Documents Retrieved: {vs.get('documents_found', 0)}")
            if "best_similarity_score" in vs:
                print(f"   Best Similarity Score: {vs['best_similarity_score']:.3f}")
                print(f"   Average Similarity Score: {vs['average_similarity_score']:.3f}")
        
        print("\n" + "="*60)
    
    def save_results(self, filename="performance_test_results.json"):
        """Save results to JSON file"""
        filepath = Path(__file__).parent / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"📁 Results saved to: {filepath}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 Starting Comprehensive Performance Test Suite")
        print("="*60)
        
        # Capture configuration
        self.capture_system_config()
        
        # Test initialization
        rag_engine = self.test_initialization_performance()
        
        # Test vector store
        self.test_vector_store_performance()
        
        # Test query performance
        self.test_query_performance(rag_engine)
        
        # Analyze confidence
        self.analyze_confidence_distribution()
        
        # Generate report
        self.generate_report()
        
        # Save results
        self.save_results()
        
        print("\n🎉 Performance test suite completed successfully!")
        return self.results

if __name__ == "__main__":
    try:
        test_suite = PerformanceTestSuite()
        results = test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
