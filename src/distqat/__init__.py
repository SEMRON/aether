import os

# This is required to register the models
if not os.environ.get('distqat_skip_model_load'):
    import distqat.models
