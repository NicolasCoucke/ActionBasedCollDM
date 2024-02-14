## Analysis scripts for 'Action-based confidence sharing and collective decision making'.

The **main.py** script parses the data into a dictionary object using the **Session.py** and **Trial.py** scripts.
This dictionary can then be used for further processing in the other scripts.

Each of the following scripts produces plots or generates Excel files that can be further analysed in R: 
- **GeneratePlots.py**
- **IndividualPerformance.py** (uses **type_2_ROC.py**)
- **InteractiveTrajectoryAnalysis.py**
- **PerformanceAnalysis**
- **ROC_time_plots.py** (uses **type_2_ROC.py**)
- **TrajectoryAnalysis** (uses data produced with **TrajectoryMatrices.py**)
