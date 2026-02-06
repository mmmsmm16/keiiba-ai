
import sys
import os
sys.path.append(os.path.abspath('.'))
from src.preprocessing.feature_pipeline import FeaturePipeline
pipeline = FeaturePipeline()
print(list(pipeline.registry.keys()))
