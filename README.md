# OPENVINO를 이용한 human counting

## 구성
- raspberry pi 3
- Neural Compute Stick 2 
- webcam

## requirement
- openvino
- opencv 4.0.1


## 필독!!

- [[라즈베리파이에 openvino 사용하기](https://jjeamin.github.io/pi/2019/03/08/NCS2/)]
- [[파일 변환](https://jjeamin.github.io/pi/2019/03/08/NCS2_IR/)]
- [[파일 변환 - YOLO](https://jjeamin.github.io/yolo,pi/2019/03/30/yolo-openvino/)]

## pd_convert

- [https://github.com/mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)

## ir 

- `.xml`
- `.bin`

## weight

- `weight file` -> `pd file` -> `.xml`,`.bin`

## Citation

### YOLO :

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
    
