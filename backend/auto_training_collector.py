#!/usr/bin/env python3
"""
Automated Training Data Collection
Runs daily to collect new data and retrain models when needed
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.data_collector import DataCollector
from training.model_trainer import HighlanderModelTrainer

class AutoTrainingCollector:
    """Automated training data collection and model retraining"""
    
    def __init__(self):
        self.basedir = Path(__file__).parent
        self.db_path = self.basedir / "instance" / "media_analysis.db"
        self.collector = DataCollector(db_path=str(self.db_path))
        self.last_run_file = self.basedir / "training" / "last_auto_collection.json"
        
    def should_collect(self) -> bool:
        """Check if we should run collection (daily)"""
        if not self.last_run_file.exists():
            return True
            
        try:
            with open(self.last_run_file, 'r') as f:
                last_run = json.load(f)
            
            last_date = datetime.fromisoformat(last_run['timestamp'])
            return datetime.now() - last_date > timedelta(days=1)
        except:
            return True
    
    def collect_and_analyze(self):
        """Collect new data and determine if retraining is needed"""
        print("🔄 Starting automated data collection...")
        
        # Collect all data
        stats = self.collector.collect_all_data()
        
        # Record this run
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'total_examples': sum(stats.values()) - stats.get('total_tokens', 0)
        }
        
        # Save run info
        self.last_run_file.parent.mkdir(exist_ok=True)
        with open(self.last_run_file, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"✅ Collection complete: {stats}")
        
        # Check if we should retrain
        total_examples = run_info['total_examples']
        if total_examples >= 30:  # Minimum for retraining
            print(f"🎯 {total_examples} examples available - sufficient for retraining")
            return True
        else:
            print(f"⏳ {total_examples} examples - need more data for retraining")
            return False
    
    def auto_retrain(self):
        """Automatically retrain model if enough new data"""
        print("🤖 Starting automated model retraining...")
        
        try:
            trainer = HighlanderModelTrainer()
            
            # Quick training for incremental updates
            success = trainer.train_model(
                output_dir="models/auto_trained",
                num_epochs=3,  # Fewer epochs for incremental training
                batch_size=2,
                learning_rate=2e-5
            )
            
            if success:
                print("✅ Automated retraining successful!")
                # TODO: Could add automatic deployment here
                return True
            else:
                print("❌ Automated retraining failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in automated retraining: {e}")
            return False
    
    def run(self):
        """Main automated collection and training pipeline"""
        if not self.should_collect():
            print("⏭️  Skipping - already ran today")
            return
        
        try:
            # Collect data
            should_retrain = self.collect_and_analyze()
            
            # Retrain if enough data
            if should_retrain:
                self.auto_retrain()
            
            print("🎉 Automated pipeline complete!")
            
        except Exception as e:
            print(f"❌ Error in automated pipeline: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    collector = AutoTrainingCollector()
    collector.run()


