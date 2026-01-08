import cv2

# 전역 변수 대신 함수 내에서 처리하거나 클로저 사용, 
# 여기서는 간단히 전역 리스트를 사용하되 매 호출마다 초기화합니다.
_points = []

def _mouse_callback(event, x, y, flags, param):
    global _points
    frame = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))
        print(f"Point {len(_points)}: {(x, y)}")
        # 클릭 지점 시각화
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        
        # 선 그리기 (다각형 모양 확인용)
        if len(_points) > 1:
            cv2.line(frame, _points[-2], _points[-1], (0, 255, 0), 2)
        if len(_points) > 2:
            # 닫힌 도형 느낌을 위해 시작점과 연결 (선택사항)
            # cv2.line(frame, _points[-1], _points[0], (0, 255, 0), 2)
            pass
            
        cv2.imshow("Pick ROI (Clockwise) -> Press 'q' to Finish", frame)

def get_roi_points(video_path):
    """
    영상을 열어 첫 프레임을 보여주고, 사용자가 ROI를 클릭하게 한 뒤 좌표를 반환합니다.
    """
    global _points
    _points = []  # 초기화

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"영상을 읽을 수 없습니다: {video_path}")

    print("\n[ROI Selection Mode]")
    print("1. 장애인 주차칸의 꼭짓점을 시계방향으로 클릭하세요.")
    print("2. 선택이 끝나면 키보드 'q'를 누르세요.\n")

    window_name = "Pick ROI (Clockwise) -> Press 'q' to Finish"
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, _mouse_callback, param=frame)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    
    print(f"Selected ROI Points: {_points}")
    return _points

# 단독 실행 시 테스트 가능
if __name__ == "__main__":
    p = get_roi_points("inputs/test.mp4")
    print("Result:", p)