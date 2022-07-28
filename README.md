# TaTic
 The code and dataset of Tatic.
 The workflow of TaTic is:
 Firstly, easy flow modeling;
 Secondly, hard flow modeling;
 Thirdly, easy-hard classification with `needdata` from easy flow modeling and `save_models` from hard flow modeling.


## Input Data Format
Each flow in the input data of Tatic should follow the format below:
>flow_identifier \t \[packet_length<sub>1</sub>, window_size<sub>1</sub>, time_interval<sub>1</sub>, packet_length<sub>2</sub>, window_size<sub>2</sub>, time_interval<sub>2</sub>, ..., packet_length<sub>H</sub>, window_size<sub>H</sub>, time_interval<sub>H</sub>\] \t flow_label

`H` denotes the length of each flow sample.
