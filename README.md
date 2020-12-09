# ACTING (Affect Control Theory based simulation of Interaction in Networked Groups)

This agent-based simulation model for group interaction is rooted in social psychological theory. The
model integrates affect control theory with networked interaction structures and sequential behavior protocols as they are often encountered in task groups. By expressing status hierarchy through network structure we build a bridge between expectation states theory and affect control theory, and are able to reproduce central results from the expectation states research program in sociological social psychology. Furthermore, we demonstrate how the model can be applied to analyze specialized task groups or sub-cultural domains by combining it with empirical data sources. As an example, we simulate groups of open-source software developers and analyze how cultural expectations influence the occupancy of high status positions in these groups.

To reproduce our results, create a virtual environment and install the same library versions:

```bash
python3 -m venv acting-env
source acting-env/bin/activate
pip install -r requierements.txt
```



You should now be able to run the jupyter notebooks that include simulations to generate the results presented in the manuscript "Modeling interaction in collaborative groups: Affect control within social structure" which we submitted to the Journal of Artificial Societies and Social Simulation.

