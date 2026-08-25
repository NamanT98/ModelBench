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
    
    if "  path: datasets/spider" not in content:
        content = content.replace(
            "dataset:\n  name: spider\n",
            "dataset:\n  name: spider\n  path: datasets/spider\n"
        )
        
        with open(file, "w") as f:
            f.write(content)
    
    print(f"Updated {file}")
