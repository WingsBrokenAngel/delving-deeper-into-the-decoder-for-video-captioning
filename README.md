# Delving Deeper into the Decoder for Video Captioning
This repository is the source code for the paper named *Delving Deeper into the Decoder for Video Captioning*.  
The paper is under review.

## Requirement
1. Python 3.6
2. TensorFlow-GPU 1.13
3. pycocoevalcap

---
## <a name="results"></a> Results

### <a name="cy"></a> Comparison on Youtube2Text

| Model   | B-4      | C        | M        | R        |  Overall |
| :------ | :------: | :------: | :------: | :------: | :------: |
|SCN      | 51.1     | 77.7     | 33.5     | -        | -        |
|MTVC     | 54.5     | 92.4     | 36.0     | 72.8     | 0.862    |
|CIDEnt-RL| 54.4     | 88.6     | 34.9     | 72.2     | 0.844    |
|HATT     | 52.9     | 73.8     | 33.8     | -        | -        |
|ECO      | 53.5     | 85.8     | 35.0     | -        | -        |
|GRU-EVE  | 47.9     | 78.1     | 35.0     | 71.5     | 0.795    |
|MARN     | 48.6     | 92.2     | 35.1     | 71.9     | 0.830    |
|SAM-SS   | 61.8     |103.0     | 37.8     | 76.8     | 0.936    |
|VNS-GRU  | **64.9** |**115.0** |**41.1**  |**78.5**  | **1.000**|

### <a name="cm"></a> Comparison on MSR-VTT

| Model       | B-4      | C        | M        | R        |  Overall |
| :------     | :------: | :------: | :------: | :------: | :------: |
|v2t_navigator| 40.8     | 44.8     | 28.2     | 60.9     | 0.917    |
|Aalto        | 39.8     | 45.7     | 26.9     | 59.8     | 0.900    |
|VideoLAB     | 39.1     | 44.1     | 27.7     | 60.6     | 0.899    |
|CIDEnt-RL    | 40.5     | 51.7     | 28.4     | 61.4     | 0.952    |
|HACA         | 43.4     | 49.7     | **29.5** | 61.8     | 0.969    |
|HATT         | 41.2     | 44.7     | 28.5     | 60.7     | 0.920    |
|GRU-EVE      | 38.2     | 48.1     | 28.4     | 60.7     | 0.919    |
|MARN         | 40.4     | 47.1     | 28.1     | 60.7     | 0.924    |
|TAMoE        | 42.2     | 48.9     | 29.4     | 62.0     | 0.958    |
|SAM-SS       | 43.8     | 51.4     | 28.9     | 62.4     | 0.977    |
|VNS-GRU      | **46.0** | **52.0** | **29.5** | **63.3** | **1.000**|