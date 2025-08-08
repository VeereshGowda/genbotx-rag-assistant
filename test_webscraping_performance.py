"""
Web Scraping Performance Test for GenBotX
Tests Wikipedia scraping and content management functionality
"""

import sys
import time
import json
import platform
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.wikipedia_scraper import WikipediaScraper
from src.content_manager import ContentManager

class WebScrapingTester:
    def __init__(self):
        self.test_results = {}
        self.test_urls = [
            "https://en.wikipedia.org/wiki/Kuru_kingdom",
            "https://en.wikipedia.org/wiki/Transformers_(film_series)",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",  # Additional test URL
            "https://en.wikipedia.org/wiki/Artificial_intelligence"  # Additional test URL
        ]
        
    def test_url_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to Wikipedia URLs"""
        print("🌐 Testing URL Connectivity...")
        
        connectivity_results = {
            "timestamp": datetime.now().isoformat(),
            "total_urls_tested": len(self.test_urls),
            "successful_connections": 0,
            "failed_connections": 0,
            "average_response_time": 0,
            "url_details": []
        }
        
        total_response_time = 0
        
        for url in self.test_urls:
            start_time = time.time()
            try:
                response = requests.get(url, timeout=10)
                response_time = time.time() - start_time
                total_response_time += response_time
                
                if response.status_code == 200:
                    connectivity_results["successful_connections"] += 1
                    status = "success"
                    print(f"   ✅ {url} - {response.status_code} ({response_time:.2f}s)")
                else:
                    connectivity_results["failed_connections"] += 1
                    status = "failed"
                    print(f"   ❌ {url} - {response.status_code} ({response_time:.2f}s)")
                
                connectivity_results["url_details"].append({
                    "url": url,
                    "status": status,
                    "status_code": response.status_code,
                    "response_time": round(response_time, 3),
                    "content_length": len(response.content) if status == "success" else 0
                })
                
            except Exception as e:
                response_time = time.time() - start_time
                connectivity_results["failed_connections"] += 1
                connectivity_results["url_details"].append({
                    "url": url,
                    "status": "error",
                    "error": str(e),
                    "response_time": round(response_time, 3)
                })
                print(f"   ❌ {url} - Error: {str(e)}")
        
        if connectivity_results["total_urls_tested"] > 0:
            connectivity_results["average_response_time"] = round(
                total_response_time / connectivity_results["total_urls_tested"], 3
            )
            connectivity_results["success_rate"] = round(
                connectivity_results["successful_connections"] / connectivity_results["total_urls_tested"] * 100, 1
            )
        
        print(f"   📊 Success Rate: {connectivity_results['success_rate']}%")
        print(f"   ⏱️  Average Response Time: {connectivity_results['average_response_time']}s")
        
        return connectivity_results
    
    def test_wikipedia_scraper_performance(self) -> Dict[str, Any]:
        """Test Wikipedia scraper functionality and performance"""
        print("\n📖 Testing Wikipedia Scraper Performance...")
        
        # Initialize scraper with test directory
        test_docs_dir = Path("test_scraped_docs")
        test_docs_dir.mkdir(exist_ok=True)
        
        scraper = WikipediaScraper(documents_folder=str(test_docs_dir))
        
        scraper_results = {
            "timestamp": datetime.now().isoformat(),
            "total_urls_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_scraping_time": 0,
            "average_scrape_time": 0,
            "total_content_size": 0,
            "average_content_size": 0,
            "scraping_speed_chars_per_sec": 0,
            "scrape_details": []
        }
        
        total_start_time = time.time()
        
        # Test with first 2 URLs to avoid overwhelming the system
        test_subset = self.test_urls[:2]
        
        for url in test_subset:
            scraper_results["total_urls_scraped"] += 1
            start_time = time.time()
            
            try:
                # Extract title for testing
                title = scraper.extract_title_from_url(url)
                print(f"   📝 Scraping: {title}")
                
                # Scrape the page
                content = scraper.scrape_wikipedia_page(url)
                scrape_time = time.time() - start_time
                
                if content and content.get('content'):
                    scraper_results["successful_scrapes"] += 1
                    content_size = len(content['content'])
                    scraper_results["total_content_size"] += content_size
                    
                    # Save content for testing
                    safe_filename = scraper.get_safe_filename(title, "test")
                    scraper.save_as_txt(content, f"{safe_filename}.txt")
                    
                    scraper_results["scrape_details"].append({
                        "url": url,
                        "title": content['title'],
                        "status": "success",
                        "scrape_time": round(scrape_time, 3),
                        "content_size": content_size,
                        "summary_length": len(content.get('summary', '')),
                        "filename": f"{safe_filename}.txt"
                    })
                    
                    print(f"      ✅ Success: {content_size:,} chars in {scrape_time:.2f}s")
                    
                else:
                    scraper_results["failed_scrapes"] += 1
                    scraper_results["scrape_details"].append({
                        "url": url,
                        "status": "failed",
                        "scrape_time": round(scrape_time, 3),
                        "error": "No content extracted"
                    })
                    print(f"      ❌ Failed: No content extracted")
                
            except Exception as e:
                scrape_time = time.time() - start_time
                scraper_results["failed_scrapes"] += 1
                scraper_results["scrape_details"].append({
                    "url": url,
                    "status": "error",
                    "scrape_time": round(scrape_time, 3),
                    "error": str(e)
                })
                print(f"      ❌ Error: {str(e)}")
        
        scraper_results["total_scraping_time"] = round(time.time() - total_start_time, 3)
        
        # Calculate averages
        if scraper_results["total_urls_scraped"] > 0:
            scraper_results["average_scrape_time"] = round(
                scraper_results["total_scraping_time"] / scraper_results["total_urls_scraped"], 3
            )
            scraper_results["success_rate"] = round(
                scraper_results["successful_scrapes"] / scraper_results["total_urls_scraped"] * 100, 1
            )
        
        if scraper_results["successful_scrapes"] > 0:
            scraper_results["average_content_size"] = round(
                scraper_results["total_content_size"] / scraper_results["successful_scrapes"]
            )
        
        if scraper_results["total_scraping_time"] > 0:
            scraper_results["scraping_speed_chars_per_sec"] = round(
                scraper_results["total_content_size"] / scraper_results["total_scraping_time"]
            )
        
        print(f"   📊 Scraping Success Rate: {scraper_results['success_rate']}%")
        print(f"   ⏱️  Total Scraping Time: {scraper_results['total_scraping_time']}s")
        print(f"   📄 Average Content Size: {scraper_results['average_content_size']:,} characters")
        print(f"   🚀 Scraping Speed: {scraper_results['scraping_speed_chars_per_sec']:,} chars/sec")
        
        return scraper_results
    
    def test_content_manager_performance(self) -> Dict[str, Any]:
        """Test content manager web scraping functionality"""
        print("\n📋 Testing Content Manager Web Scraping...")
        
        content_manager = ContentManager(
            documents_folder="test_content_docs",
            upload_folder="test_uploads"
        )
        
        content_results = {
            "timestamp": datetime.now().isoformat(),
            "urls_processed": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "total_content_scraped": 0,
            "average_content_size": 0,
            "process_details": []
        }
        
        total_start_time = time.time()
        
        # Test with first 2 URLs
        test_subset = self.test_urls[:2]
        
        for url in test_subset:
            content_results["urls_processed"] += 1
            start_time = time.time()
            
            try:
                print(f"   🔄 Processing URL: {url}")
                
                # Test URL scraping through content manager
                result = content_manager.scrape_webpage(url)
                process_time = time.time() - start_time
                
                if result and result.get('content'):
                    content_results["successful_processes"] += 1
                    content_size = len(result.get('content', ''))
                    content_results["total_content_scraped"] += content_size
                    
                    content_results["process_details"].append({
                        "url": url,
                        "status": "success",
                        "processing_time": round(process_time, 3),
                        "content_length": content_size,
                        "title": result.get('title', ''),
                        "scraped_at": result.get('scraped_at', '')
                    })
                    
                    print(f"      ✅ Success: {content_size:,} chars in {process_time:.2f}s")
                    print(f"         Title: {result.get('title', 'Unknown')}")
                    
                else:
                    content_results["failed_processes"] += 1
                    content_results["process_details"].append({
                        "url": url,
                        "status": "failed",
                        "processing_time": round(process_time, 3),
                        "error": 'No content extracted or result returned'
                    })
                    print(f"      ❌ Failed: No content extracted or result returned")
                
            except Exception as e:
                process_time = time.time() - start_time
                content_results["failed_processes"] += 1
                content_results["process_details"].append({
                    "url": url,
                    "status": "error",
                    "processing_time": round(process_time, 3),
                    "error": str(e)
                })
                print(f"      ❌ Error: {str(e)}")
        
        content_results["total_processing_time"] = round(time.time() - total_start_time, 3)
        
        # Calculate averages
        if content_results["urls_processed"] > 0:
            content_results["average_processing_time"] = round(
                content_results["total_processing_time"] / content_results["urls_processed"], 3
            )
            content_results["success_rate"] = round(
                content_results["successful_processes"] / content_results["urls_processed"] * 100, 1
            )
        
        if content_results["successful_processes"] > 0:
            content_results["average_content_size"] = round(
                content_results["total_content_scraped"] / content_results["successful_processes"]
            )
        
        print(f"   📊 Processing Success Rate: {content_results['success_rate']}%")
        print(f"   ⏱️  Average Processing Time: {content_results['average_processing_time']}s")
        print(f"   � Total Content Scraped: {content_results.get('total_content_scraped', 0):,} characters")
        
        return content_results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all web scraping tests and compile results"""
        print("🚀 Starting Comprehensive Web Scraping Performance Test")
        print("=" * 60)
        
        test_start_time = time.time()
        
        # Run all tests
        connectivity_results = self.test_url_connectivity()
        scraper_results = self.test_wikipedia_scraper_performance()
        content_manager_results = self.test_content_manager_performance()
        
        total_test_time = time.time() - test_start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_test_duration": round(total_test_time, 3),
                "python_version": platform.python_version(),
                "operating_system": f"{platform.system()} {platform.release()}"
            },
            "connectivity_test": connectivity_results,
            "wikipedia_scraper_test": scraper_results,
            "content_manager_test": content_manager_results,
            "summary_metrics": {
                "overall_url_success_rate": round(
                    (connectivity_results.get("success_rate", 0) + 
                     scraper_results.get("success_rate", 0) + 
                     content_manager_results.get("success_rate", 0)) / 3, 1
                ),
                "total_content_scraped_chars": scraper_results.get("total_content_size", 0),
                "average_scraping_speed_chars_per_sec": scraper_results.get("scraping_speed_chars_per_sec", 0),
                "total_processing_time": round(
                    connectivity_results.get("average_response_time", 0) +
                    scraper_results.get("total_scraping_time", 0) +
                    content_manager_results.get("total_processing_time", 0), 3
                )
            }
        }
        
        # Save results
        results_file = f"webscraping_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 WEB SCRAPING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"\n🌐 CONNECTIVITY:")
        print(f"   • URL Success Rate: {connectivity_results.get('success_rate', 0)}%")
        print(f"   • Average Response Time: {connectivity_results.get('average_response_time', 0)}s")
        
        print(f"\n📖 WIKIPEDIA SCRAPER:")
        print(f"   • Scraping Success Rate: {scraper_results.get('success_rate', 0)}%")
        print(f"   • Total Content Scraped: {scraper_results.get('total_content_size', 0):,} characters")
        print(f"   • Scraping Speed: {scraper_results.get('scraping_speed_chars_per_sec', 0):,} chars/sec")
        print(f"   • Average Scrape Time: {scraper_results.get('average_scrape_time', 0)}s")
        
        print(f"\n📋 CONTENT MANAGER:")
        print(f"   • Processing Success Rate: {content_manager_results.get('success_rate', 0)}%")
        print(f"   • Average Processing Time: {content_manager_results.get('average_processing_time', 0)}s")
        print(f"   • Total Content Scraped: {content_manager_results.get('total_content_scraped', 0):,} characters")
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        summary = comprehensive_results["summary_metrics"]
        print(f"   • Overall Success Rate: {summary['overall_url_success_rate']}%")
        print(f"   • Total Test Duration: {comprehensive_results['test_metadata']['total_test_duration']}s")
        print(f"   • Content Processing Speed: {summary['average_scraping_speed_chars_per_sec']:,} chars/sec")
        
        print(f"\n📁 Results saved to: {results_file}")
        print("=" * 60)
        
        return comprehensive_results

def main():
    """Main function to run web scraping performance tests"""
    tester = WebScrapingTester()
    results = tester.run_comprehensive_test()
    print("\n🎉 Web scraping performance test completed successfully!")
    return results

if __name__ == "__main__":
    main()
