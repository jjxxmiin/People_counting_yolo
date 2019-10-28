# OPENVINO를 이용한 human counting

[POST](https://jjeamin.github.io/yolo/2019/03/28/yolo_counting/)

# 사용법

1. 위에 xml_path와 bin_path를 xml,bin 파일 경로로 바꾸어준다.
2. capture function에서 아래 코드 주석처리
```
# cv2.imwrite('test2.jpg',frame)
```
3. 스케쥴러를 이용해서 웹캠으로 10초에 한번씩 사람의 수를 counting한다.

```
python3 main.py
```

**logger**
1. 인터넷이 연결되어있는지 확인 -> 서버로 전송가능
2. 카메라가 연결되어있는지 확인

---

# Citation

**YOLO**

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

**SORT**

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


# Reference
- [https://github.com/guillelopez/python-traffic-counter-with-yolo-and-sort](https://github.com/guillelopez/python-traffic-counter-with-yolo-and-sort)
- [https://github.com/jjeamin/OpenVINO-YoloV3](https://github.com/jjeamin/OpenVINO-YoloV3)
- [https://github.com/abewley/sort](https://github.com/abewley/sort)
