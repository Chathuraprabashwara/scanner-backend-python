import asyncio
import websockets
import base64
import cv2
import numpy as np


async def handler(websocket):
    async for message in websocket:
        try:
            # Remove base64 prefix
            header, encoded = message.split(",", 1)
            img_data = base64.b64decode(encoded)

            # Convert to OpenCV image
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # --- Detect edges for rectangle ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding rectangle
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 1000:  # ignore small noise
                    x, y, w, h = cv2.boundingRect(largest)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show on Python window (optional)
            # cv2.imshow("Document Scanner", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Optional: send processed image back to React
            _, buffer = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            await websocket.send(f"data:image/jpeg;base64,{b64_frame}")

        except Exception as e:
            print("Error:", e)


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8080):
        print("WebSocket server running on port 8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())