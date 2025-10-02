#!/usr/bin/env python3
"""
Training Data Inspector

This script helps you inspect and understand what training data has been collected.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def inspect_training_data(model_name='mediamap'):
    """Inspect collected training data for a specific model"""
    
    print(f"🔍 Training Data Inspector for {model_name.upper()}")
    print("=" * 60)
    
    # Define paths
    base_dir = Path("training_data") / model_name
    
    if not base_dir.exists():
        print(f"❌ No training data found for {model_name}")
        print(f"   Expected location: {base_dir}")
        print("   Run data collection first from the training page.")
        return
    
    print(f"📁 Data Location: {base_dir.absolute()}")
    print()
    
    # Check conversations
    print("💬 CONVERSATIONS:")
    conv_dir = base_dir / "conversations"
    if conv_dir.exists():
        for conv_file in conv_dir.glob("*.json"):
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    count = len(data)
                    print(f"   ✅ {conv_file.name}: {count} conversations")
                    
                    # Show sample if data exists
                    if count > 0 and data[0]:
                        sample = data[0]
                        if isinstance(sample, dict):
                            input_preview = str(sample.get('input', sample.get('message', '')))[:100]
                            print(f"      Sample: {input_preview}...")
                else:
                    print(f"   ⚠️  {conv_file.name}: Invalid format")
                    
            except Exception as e:
                print(f"   ❌ {conv_file.name}: Error reading - {e}")
    else:
        print("   📂 No conversations directory found")
    
    print()
    
    # Check quality reports
    print("📊 QUALITY REPORTS:")
    quality_dir = base_dir / "quality_reports"
    if quality_dir.exists():
        reports = list(quality_dir.glob("*.json"))
        if reports:
            # Get the latest report
            latest_report = max(reports, key=lambda x: x.stat().st_mtime)
            
            try:
                with open(latest_report, 'r') as f:
                    report = json.load(f)
                
                print(f"   📄 Latest Report: {latest_report.name}")
                print(f"   📅 Date: {report.get('collection_date', 'Unknown')}")
                print(f"   📈 Quality Score: {report.get('quality_metrics', {}).get('quality_score', 0):.2f}/1.0")
                print(f"   📊 Total Examples: {report.get('quality_metrics', {}).get('total_examples', 0)}")
                
                # Show data sources
                sources = report.get('data_sources', {})
                print("   📋 Data Sources:")
                for source, count in sources.items():
                    if count > 0:
                        print(f"      • {source.replace('_', ' ').title()}: {count}")
                
                # Show recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    print("   💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show top 3
                        print(f"      • {rec}")
                        
            except Exception as e:
                print(f"   ❌ Error reading latest report: {e}")
        else:
            print("   📂 No quality reports found")
    else:
        print("   📂 No quality reports directory found")
    
    print()
    
    # Check PDF documents
    print("📄 PDF DOCUMENTS:")
    pdf_dir = base_dir / "pdfs"
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        txt_files = list(pdf_dir.glob("*.txt"))
        
        if pdf_files or txt_files:
            print(f"   📁 {len(pdf_files)} PDF files, {len(txt_files)} extracted text files")
            
            # Show latest PDF
            if pdf_files:
                latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest_pdf.stat().st_mtime)
                file_size = latest_pdf.stat().st_size
                print(f"   📄 Latest PDF: {latest_pdf.name}")
                print(f"      Size: {file_size:,} bytes, Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
                
                # Check if text was extracted
                txt_equivalent = pdf_dir / latest_pdf.name.replace('.pdf', '.txt')
                if txt_equivalent.exists():
                    with open(txt_equivalent, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    print(f"      Extracted text: {len(text_content):,} characters")
        else:
            print("   📂 No PDF files uploaded yet")
            print("   💡 Use the 'Upload PDF' button on the training page to add documents")
    else:
        print("   📂 PDF directory not found")
    
    print()
    
    # Check other data types
    data_types = [
        ("research", "🔬 Research Papers"),
        ("feedback", "💬 User Feedback"),
        ("continuous_learning", "🧠 Continuous Learning"),
        ("performance_monitoring", "📈 Performance Data")
    ]
    
    for dir_name, display_name in data_types:
        data_dir = base_dir / dir_name
        if data_dir.exists():
            files = list(data_dir.glob("*.json"))
            if files:
                print(f"{display_name}: {len(files)} files")
                # Show latest file info
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    print(f"   Latest: {latest_file.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"{display_name}: Empty")
        else:
            print(f"{display_name}: Not found")
    
    print()
    
    # Check synthetic data
    synthetic_file = base_dir / "synthetic_training_data.json"
    if synthetic_file.exists():
        try:
            with open(synthetic_file, 'r') as f:
                synthetic_data = json.load(f)
            print(f"🤖 SYNTHETIC DATA: {len(synthetic_data)} examples")
            if synthetic_data:
                sample = synthetic_data[0]
                input_preview = str(sample.get('input', ''))[:100]
                print(f"   Sample: {input_preview}...")
        except Exception as e:
            print(f"🤖 SYNTHETIC DATA: Error reading - {e}")
    else:
        print("🤖 SYNTHETIC DATA: Not found")
    
    print()
    print("=" * 60)
    print("💡 TIP: To collect more data, use the 'Collect Data' button")
    print("   on the training page at http://localhost:3000/admin/training")

def main():
    """Main function"""
    import sys
    
    model_name = 'mediamap'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    inspect_training_data(model_name)
    
    print(f"\n🔍 Want to inspect {model_name} data in detail?")
    print(f"   Files are located in: training_data/{model_name}/")
    print("   You can open the JSON files in any text editor to see the full content.")

if __name__ == "__main__":
    main()
