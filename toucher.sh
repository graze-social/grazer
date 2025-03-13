mkdir -p tests/app/algos/operators
mkdir -p tests/app/models
mkdir -p tests/app/ray
mkdir -p tests/app/runpod

# 2) Create __init__.py files so the tests are treated as a package
touch tests/__init__.py
touch tests/app/__init__.py
touch tests/app/algos/__init__.py
touch tests/app/algos/operators/__init__.py
touch tests/app/models/__init__.py
touch tests/app/ray/__init__.py
touch tests/app/runpod/__init__.py

# 3) Create test files for each top-level module in app/
for module in algorithm_asset_cacher egress helpers jetstream logger logic_evaluator redis settings; do
  touch "tests/app/test_${module}.py"
done

# 4) Create test files for algos submodules
for module in base manager; do
  touch "tests/app/algos/test_${module}.py"
done

# 5) Create test files for algos/operators submodules
for module in attribute combo emotion_sentiment entity financial_sentiment huggingface_classifier \
               image_arbitrary image_nsfw language moderation regex sentiment social \
               text_arbitrary topic toxicity transformer
do
  touch "tests/app/algos/operators/test_${module}.py"
done

# 6) Create test files for models submodules
for module in huggingface_classifier image_arbitrary_classifier image_nsfw_classifier ml_model \
               text_arbitrary_classifier text_embedder
do
  touch "tests/app/models/test_${module}.py"
done

# 7) Create test files for ray submodules
for module in cache cpu_worker dispatcher gpu_worker network_worker semaphore timing_base utils
do
  touch "tests/app/ray/test_${module}.py"
done

# 8) Create test files for runpod submodules
for module in auditor backfiller backtester base processor router
do
  touch "tests/app/runpod/test_${module}.py"
done
