import argparse

import cv2
import numpy as np
import tensorflow as tf

import data
import model

def main(checkpoints_dir):

    #TODO: load model
    net = model.Baseline(num_labels=6)
    latest = tf.train.latest_checkpoint(checkpoints_dir)
    net.load_weights(latest)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Get a prediction from your model
        formated_image = data.preprocess_image(frame)
        prediction = net.predict(formated_image[None, ...])
        prediction = str(np.argmax(prediction[0, ...]))

        # Write prediction on the frame
        cv2.putText(frame, prediction, (10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run the application")
    parser.add_argument('--checkpoints-dir', default='checkpoints')

    args = parser.parse_args()
    print(args)

    main(args.checkpoints_dir)