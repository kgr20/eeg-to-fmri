import os

def replace_np_bool(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                content = content.replace('np.bool__', 'np.bool_')
                with open(filepath, 'w') as f:
                    f.write(content)

# Use the path printed by find_tfp_path.py
tfp_path = "/Users/apple/anaconda3/envs/eeg_fmri_env38/lib/python3.8/site-packages/tensorflow_probability"
replace_np_bool(tfp_path)
print("Done")