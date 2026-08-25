import os

files = [
    "configs/m7_hybrid_score_alpha25.yaml",
    "configs/m7_hybrid_score_alpha50.yaml",
    "configs/m7_hybrid_score_alpha75.yaml",
    "configs/m7_hybrid_rrf.yaml",
    "configs/m7_hybrid_union.yaml"
]

for file in files:
    with open(file, "r") as f:
        content = f.read()
    
    if "batch_size: 16" in content:
        content = content.replace("batch_size: 16", "batch_size: 8")
        
        with open(file, "w") as f:
            f.write(content)
        print(f"Updated {file} to batch_size 8")
    else:
        print(f"No changes needed for {file} (might already be updated)")
