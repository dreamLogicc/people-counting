import cv2
import imageio.v3 as iio
from pathlib import Path
from cvlib.object_detection import draw_bbox
import cvlib as cv


def play(bytes):

    def get_bbox_center(bbox):
        x,y,w,h = bbox
        return (int((x+w)/2), int((y+h)/2))

    frames_count = 0
    people_count = 0

    for frame in iio.imiter(bytes, format_hint=".mp4"):

        frames_count = frames_count + 1
        if frames_count % 6 != 0:
            continue

        frame_temp = cv2.resize(frame, (600,600))

        cv2.line(img=frame_temp, pt1=(280, 0), pt2=(280, 600),
                color=(255, 0, 0), lineType=8, thickness=1)
        
        cv2.line(img=frame_temp, pt1=(320, 0), pt2=(320, 600),
                color=(255, 0, 0), lineType=8, thickness=1)
        
        bbox, labels, conf = cv.detect_common_objects(frame_temp)
        frame_temp = draw_bbox(frame_temp, bbox=bbox, labels=labels, confidence=conf)
        
        for (label, bb)  in zip(labels, bbox):
            if (label == 'person' or label == 'bicycle') and (280 <= get_bbox_center(bb)[0] <= 320):
                people_count = people_count + 1
            labels.remove(label)
            bbox.remove(bb)

        cv2.putText(frame_temp, f'people: {people_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)   

        cv2.imshow('Video', cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    bytes = Path("./video_1.mp4").read_bytes()
    play(bytes)