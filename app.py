import cv2
from ultralytics import YOLO


# ============================================================
# SMART CLASSROOM - AI EXAM PROCTORING
# ============================================================

print("Loading AI models...")


# ============================================================
# LOAD PHONE MODEL
# ============================================================

# Pretrained YOLO model
# COCO class 67 = cell phone

phone_model = YOLO("yolo26n.pt")


# ============================================================
# FACE DETECTORS
# ============================================================

# Frontal face detector
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Profile face detector
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_profileface.xml"
)


if frontal_cascade.empty():

    print("ERROR: Frontal face detector could not be loaded.")
    exit()


if profile_cascade.empty():

    print("ERROR: Profile face detector could not be loaded.")
    exit()


# ============================================================
# OPEN LAPTOP WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open laptop webcam.")
    exit()


# Camera resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# STUDENT TRACKING
# ============================================================

next_student_id = 1

students = {}


# Number of frames a student must be turned
# before we call it "Looking around"

TURN_CONFIRM_FRAMES = 5


# ============================================================
# WINDOW
# ============================================================

window_name = "Smart Classroom - AI Exam Proctoring"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)


# ============================================================
# START MESSAGE
# ============================================================

print()
print("==============================================")
print(" Smart Classroom - AI Exam Proctoring")
print("==============================================")
print()
print("Camera      : Laptop Webcam")
print("Face        : Enabled")
print("Head Turn   : Enabled")
print("Phone       : Enabled")
print("Tracking    : Enabled")
print()
print("Eye movement will NOT trigger Looking Around.")
print("A clear side/profile face is required.")
print()
print("Press Q to quit.")
print()


# ============================================================
# HELPER FUNCTION
# ============================================================

def center_distance(point1, point2):

    return (
        (
            (point1[0] - point2[0]) ** 2
            +
            (point1[1] - point2[1]) ** 2
        )
        ** 0.5
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Could not read webcam frame.")
        break


    annotated_frame = frame.copy()


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Improve contrast

    gray = cv2.equalizeHist(gray)


    # ========================================================
    # FRONTAL FACE DETECTION
    # ========================================================

    frontal_faces = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(60, 60)
    )


    # ========================================================
    # PROFILE FACE DETECTION
    # ========================================================

    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )


    # --------------------------------------------------------
    # DETECT PROFILE ON OTHER SIDE
    # --------------------------------------------------------

    flipped_gray = cv2.flip(
        gray,
        1
    )


    flipped_profiles = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )


    # Convert flipped coordinates back

    image_width = frame.shape[1]


    for (x, y, w, h) in flipped_profiles:

        original_x = image_width - x - w

        profile_faces = list(profile_faces) + [
            (
                original_x,
                y,
                w,
                h
            )
        ]


    # ========================================================
    # BUILD FACE LIST
    # ========================================================

    current_faces = []


    for (x, y, w, h) in frontal_faces:

        center_x = x + (w // 2)

        center_y = y + (h // 2)


        current_faces.append({
            "box": (x, y, w, h),
            "center": (center_x, center_y),
            "profile": False
        })


    # ========================================================
    # CHECK PROFILE FACES
    # ========================================================

    for (px, py, pw, ph) in profile_faces:

        profile_center_x = px + (pw // 2)

        profile_center_y = py + (ph // 2)


        # Check whether this profile face is already
        # represented by a frontal face

        matched = False


        for face in current_faces:

            fx, fy, fw, fh = face["box"]

            face_cx, face_cy = face["center"]


            distance = center_distance(
                (profile_center_x, profile_center_y),
                (face_cx, face_cy)
            )


            if distance < max(pw, ph):

                matched = True
                break


        # Add profile face only if it isn't
        # already represented by a frontal face

        if not matched:

            current_faces.append({
                "box": (px, py, pw, ph),
                "center": (
                    profile_center_x,
                    profile_center_y
                ),
                "profile": True
            })


    # ========================================================
    # STUDENT ID ASSIGNMENT
    # ========================================================

    # Keep track of IDs used in this frame
    # so two faces don't receive the same ID.

    used_ids = set()


    for face in current_faces:

        x, y, w, h = face["box"]

        face_cx, face_cy = face["center"]


        # ----------------------------------------------------
        # FIND CLOSEST EXISTING STUDENT
        # ----------------------------------------------------

        best_student_id = None

        best_distance = float("inf")


        for sid, student_data in students.items():

            old_center = student_data["center"]

            distance = center_distance(
                (face_cx, face_cy),
                old_center
            )


            if (
                distance < 150
                and
                distance < best_distance
                and
                sid not in used_ids
            ):

                best_distance = distance

                best_student_id = sid


        # ----------------------------------------------------
        # CREATE NEW STUDENT
        # ----------------------------------------------------

        if best_student_id is None:

            student_id = next_student_id

            next_student_id += 1

            students[student_id] = {
                "center": (
                    face_cx,
                    face_cy
                ),
                "turn_frames": 0
            }

        else:

            student_id = best_student_id


        used_ids.add(student_id)


        # Update student position

        students[student_id]["center"] = (
            face_cx,
            face_cy
        )


        # ====================================================
        # HEAD TURN DETECTION
        # ====================================================

        is_profile = face["profile"]


        if is_profile:

            students[student_id]["turn_frames"] += 1

        else:

            # Reset when face comes back
            # toward the camera

            students[student_id]["turn_frames"] = 0


        turn_frames = students[student_id]["turn_frames"]


        # ====================================================
        # STUDENT STATUS
        # ====================================================

        if turn_frames >= TURN_CONFIRM_FRAMES:

            label = "Looking around"

            box_color = (
                0,
                165,
                255
            )

        else:

            label = "No cheating"

            box_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )


        # ====================================================
        # STUDENT LABEL
        # ====================================================

        student_text = (
            f"Student {student_id}: {label}"
        )


        cv2.putText(
            annotated_frame,
            student_text,
            (
                x,
                max(y - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )


    # ========================================================
    # PHONE DETECTION
    # ========================================================

    phone_results = phone_model.predict(
        source=frame,
        classes=[67],
        conf=0.60,
        imgsz=320,
        verbose=False
    )


    phone_detected = False


    for result in phone_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # PHONE SIZE FILTER
            # ------------------------------------------------

            width = x2 - x1

            height = y2 - y1


            if width < 30 or height < 30:

                continue


            phone_detected = True


            # =================================================
            # PHONE BOX
            # =================================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )


            # =================================================
            # PHONE LABEL
            # =================================================

            phone_text = (
                f"PHONE USE {confidence:.2f}"
            )


            cv2.putText(
                annotated_frame,
                phone_text,
                (
                    x1,
                    max(y1 - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


    # ========================================================
    # CHECK IF ANY STUDENT IS LOOKING AROUND
    # ========================================================

    looking_around = False


    for student_data in students.values():

        if student_data["turn_frames"] >= TURN_CONFIRM_FRAMES:

            looking_around = True

            break


    # ========================================================
    # TOP STATUS
    # ========================================================

    if phone_detected:

        status_text = "ALERT - PHONE DETECTED"

        status_color = (
            0,
            0,
            255
        )


    elif looking_around:

        status_text = "WARNING - LOOKING AROUND"

        status_color = (
            0,
            165,
            255
        )


    else:

        status_text = "SAFE - NO CHEATING"

        status_color = (
            0,
            255,
            0
        )


    # ========================================================
    # STATUS BAR
    # ========================================================

    cv2.rectangle(
        annotated_frame,
        (0, 0),
        (
            annotated_frame.shape[1],
            55
        ),
        (30, 30, 30),
        -1
    )


    # ========================================================
    # STATUS TEXT
    # ========================================================

    cv2.putText(
        annotated_frame,
        status_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )


    # ========================================================
    # STUDENT COUNT
    # ========================================================

    student_count_text = (
        f"Students: {len(current_faces)}"
    )


    cv2.putText(
        annotated_frame,
        student_count_text,
        (
            annotated_frame.shape[1] - 180,
            38
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        window_name,
        annotated_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("Smart Classroom monitoring stopped.")