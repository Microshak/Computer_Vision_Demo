import cv2
 
scale_factor = 1
min_neighbors = 3
min_size = (50, 50)
webcam=True #if working with video file then make it 'False'
 
def detect(path):
 
    cascade = cv2.CascadeClassifier(path)
    video_cap = cv2.VideoCapture(0) # use 0,1,2..depanding on your webcam
    while True:
        # Capture frame-by-frame
        try:
            ret, img = video_cap.read()
 
            #converting to gray image for faster video processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
     
                # Display the resulting frame
                cv2.imshow('Face Detection on Video', img)
        except Exception as e:
            print(str(e))
                #wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
 
def main():
    cascadeFilePath="haarcascade_frontalface_alt.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()