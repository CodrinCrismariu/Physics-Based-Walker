import os

with open("src/hlip_clf_g1/mdp/observations.py", "w") as f:
    text = open("src/hlip_clf_g1/mdp/observations.py.backup", "r").read() if os.path.exists("src/hlip_clf_g1/mdp/observations.py.backup") else open("src/hlip_clf_g1/mdp/observations.py", "r").read()
    if "heightmap_data" not in text:
        text += """
def heightmap_data(env) -> torch.Tensor:
  return env.scene.sensors["heightmap"].data.distances
"""
    f.write(text)
