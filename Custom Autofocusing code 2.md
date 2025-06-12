```python
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from openflexure_microscope_client import MicroscopeClient

# Connect to microscope
microscope = MicroscopeClient("192.168.0.111")

# Parameters
coarse_step_size = 200
coarse_num_steps = 10
fine_step_size = 50
settle_time = 0.01  # Reduced settle time for speed
coarse_drop_threshold = 10


def compute_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def move_microscope_rel(z_delta):
    microscope.move_rel({"x": 0, "y": 0, "z": z_delta})
    time.sleep(settle_time)


def move_microscope_abs(z_target):
    microscope.move({"x": microscope.position['x'], "y": microscope.position['y'], "z": z_target})
    time.sleep(settle_time)


def capture_image():
    return np.array(microscope.grab_image())


def evaluate_focus_direction(step_size, num_steps):
    base_z = microscope.position['z']
    variances = []

    for _ in range(num_steps):
        move_microscope_rel(step_size)
        var = compute_laplacian_variance(capture_image())
        variances.append((var, microscope.position['z']))
    move_microscope_abs(base_z)

    for _ in range(num_steps):
        move_microscope_rel(-step_size)
        var = compute_laplacian_variance(capture_image())
        variances.append((var, microscope.position['z']))
    move_microscope_abs(base_z)

    best_var, best_z = max(variances, key=lambda x: x[0])
    direction = 1 if best_z > base_z else -1
    return direction


def move_until_drop(step_size, direction):
    positions = []
    variances = []
    images = []

    while True:
        move_microscope_rel(direction * step_size)
        image = capture_image()
        var = compute_laplacian_variance(image)

        positions.append(microscope.position['z'])
        variances.append(var)
        images.append(image)

        if len(variances) >= 3:
            if variances[-2] > variances[-1] and (variances[-2] - variances[-1]) > coarse_drop_threshold:
                break

    max_idx = np.argmax(variances)
    best_z = positions[max_idx]
    best_var = variances[max_idx]
    best_image = images[max_idx]

    move_microscope_abs(best_z)
    return best_z, best_var, best_image, positions, variances


def coarse_focus():
    print("Evaluating direction for coarse focus...")
    direction = evaluate_focus_direction(coarse_step_size, coarse_num_steps)
    print(f"Coarse focus direction: {'up' if direction == 1 else 'down'}")
    best_z, best_var, best_image, z_positions, variances = move_until_drop(coarse_step_size, direction)
    print(f"Coarse focus at Z={best_z} with variance={best_var:.2f}")
    return best_z, best_var, best_image, z_positions, variances


def fine_focus(coarse_z, z_positions, variances):
    print("Starting fine focus sweep...")

    start_z = coarse_z - 200
    end_z = coarse_z + 200

    best_var = -1
    best_image = None
    best_z = None

    for z in range(start_z, end_z + 1, fine_step_size):
        move_microscope_abs(z)
        image = capture_image()
        var = compute_laplacian_variance(image)
        print(f"Z={z}, Variance={var:.4f}")

        if var > best_var:
            best_var = var
            best_image = image
            best_z = z

    print(f"Fine focus complete at Z={best_z} with variance={best_var:.2f}")
    move_microscope_abs(best_z)
    return best_z, best_var, best_image


def autofocus():
    coarse_z, _, coarse_image, z_positions, variances = coarse_focus()
    fine_z, _, fine_image = fine_focus(coarse_z, z_positions, variances)
    return fine_z, fine_image


# Output directory
desktop = Path.home() / "Desktop"
output_dir = desktop / "microscope1" / "405nm testing" / "edith4"
output_dir.mkdir(parents=True, exist_ok=True)

# Store initial position
starting_pos = microscope.position.copy()

# Raster scan parameters
step_size_x = 800
step_size_y = 800
x_direction = 1
x_steps = 0
max_x_steps = 10

# Raster scan and capture 50 images
for i in range(50):
    if i > 0:
        if x_steps < max_x_steps - 1:
            starting_pos['x'] += x_direction * step_size_x
            x_steps += 1
        else:
            starting_pos['y'] += step_size_y
            x_direction *= -1
            x_steps = 0

        print(f"Moving to next position: X={starting_pos['x']}, Y={starting_pos['y']}")
        microscope.move(starting_pos)
        print(f"Position after move: {microscope.position}")

    # Autofocus at current position
    best_z, focused_image = autofocus()

    # Update Z
    starting_pos['z'] = best_z

    # Save focused image
    image_filename = output_dir / f"image_{i+1}.png"
    Image.fromarray(focused_image).save(image_filename)
    print(f"Saved image {i+1} at {image_filename}")

    # Optional: Preview
    plt.imshow(focused_image)
    plt.title(f"Image {i+1} | X: {starting_pos['x']}  Y: {starting_pos['y']}  Z: {best_z}")
    plt.axis('off')
    plt.show()

# Return to initial position
microscope.move(starting_pos)
assert microscope.position == starting_pos
print(f"Captured and saved 50 images in '{output_dir}'.")

```

    Evaluating direction for coarse focus...
    Coarse focus direction: up
    Coarse focus at Z=90880 with variance=390.39
    Starting fine focus sweep...
    Z=90680, Variance=360.6743
    Z=90730, Variance=366.4639
    Z=90780, Variance=357.9413
    Z=90830, Variance=360.4275
    Z=90880, Variance=359.9521
    Z=90930, Variance=349.3230
    Z=90980, Variance=352.1148
    Z=91030, Variance=330.4851
    Z=91080, Variance=317.3579
    Fine focus complete at Z=90730 with variance=366.46
    Saved image 1 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_1.png
    


    
![png](output_0_1.png)
    


    Moving to next position: X=128647, Y=9069
    Position after move: {'x': 128647, 'y': 9069, 'z': 90730}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=90530 with variance=591.05
    Starting fine focus sweep...
    Z=90330, Variance=413.0909
    Z=90380, Variance=425.6590
    Z=90430, Variance=441.6917
    Z=90480, Variance=464.3099
    Z=90530, Variance=475.1109
    Z=90580, Variance=527.0782
    Z=90630, Variance=566.2456
    Z=90680, Variance=596.8576
    Z=90730, Variance=609.1223
    Fine focus complete at Z=90730 with variance=609.12
    Saved image 2 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_2.png
    


    
![png](output_0_3.png)
    


    Moving to next position: X=129447, Y=9069
    Position after move: {'x': 129447, 'y': 9069, 'z': 90730}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=90530 with variance=621.15
    Starting fine focus sweep...
    Z=90330, Variance=443.3828
    Z=90380, Variance=457.1126
    Z=90430, Variance=462.4403
    Z=90480, Variance=466.7012
    Z=90530, Variance=467.7568
    Z=90580, Variance=518.3942
    Z=90630, Variance=570.8300
    Z=90680, Variance=637.8793
    Z=90730, Variance=663.6901
    Fine focus complete at Z=90730 with variance=663.69
    Saved image 3 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_3.png
    


    
![png](output_0_5.png)
    


    Moving to next position: X=130247, Y=9069
    Position after move: {'x': 130247, 'y': 9069, 'z': 90730}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=90530 with variance=514.64
    Starting fine focus sweep...
    Z=90330, Variance=455.5310
    Z=90380, Variance=455.6580
    Z=90430, Variance=458.2038
    Z=90480, Variance=458.3436
    Z=90530, Variance=459.3198
    Z=90580, Variance=491.2336
    Z=90630, Variance=501.5973
    Z=90680, Variance=511.9991
    Z=90730, Variance=497.9549
    Fine focus complete at Z=90680 with variance=512.00
    Saved image 4 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_4.png
    


    
![png](output_0_7.png)
    


    Moving to next position: X=131047, Y=9069
    Position after move: {'x': 131047, 'y': 9069, 'z': 90680}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=89680 with variance=194.75
    Starting fine focus sweep...
    Z=89480, Variance=150.8184
    Z=89530, Variance=151.3962
    Z=89580, Variance=151.0243
    Z=89630, Variance=147.4421
    Z=89680, Variance=148.1957
    Z=89730, Variance=165.5773
    Z=89780, Variance=176.7825
    Z=89830, Variance=186.0092
    Z=89880, Variance=189.4166
    Fine focus complete at Z=89880 with variance=189.42
    Saved image 5 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_5.png
    


    
![png](output_0_9.png)
    


    Moving to next position: X=131847, Y=9069
    Position after move: {'x': 131847, 'y': 9069, 'z': 89880}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=89480 with variance=213.49
    Starting fine focus sweep...
    Z=89280, Variance=184.7759
    Z=89330, Variance=183.7153
    Z=89380, Variance=182.2894
    Z=89430, Variance=181.5700
    Z=89480, Variance=183.0522
    Z=89530, Variance=185.9960
    Z=89580, Variance=187.5693
    Z=89630, Variance=204.5848
    Z=89680, Variance=218.6260
    Fine focus complete at Z=89680 with variance=218.63
    Saved image 6 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_6.png
    


    
![png](output_0_11.png)
    


    Moving to next position: X=132647, Y=9069
    Position after move: {'x': 132647, 'y': 9069, 'z': 89680}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=89280 with variance=204.19
    Starting fine focus sweep...
    Z=89080, Variance=193.1937
    Z=89130, Variance=193.2126
    Z=89180, Variance=194.0760
    Z=89230, Variance=193.8378
    Z=89280, Variance=193.4332
    Z=89330, Variance=194.4131
    Z=89380, Variance=193.8384
    Z=89430, Variance=197.1972
    Z=89480, Variance=204.3618
    Fine focus complete at Z=89480 with variance=204.36
    Saved image 7 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_7.png
    


    
![png](output_0_13.png)
    


    Moving to next position: X=133447, Y=9069
    Position after move: {'x': 133447, 'y': 9069, 'z': 89480}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    Coarse focus at Z=89080 with variance=209.63
    Starting fine focus sweep...
    Z=88880, Variance=155.1084
    Z=88930, Variance=151.7012
    Z=88980, Variance=153.6972
    Z=89030, Variance=154.9630
    Z=89080, Variance=158.1378
    Z=89130, Variance=160.8224
    Z=89180, Variance=163.3949
    Z=89230, Variance=184.8354
    Z=89280, Variance=204.6299
    Fine focus complete at Z=89280 with variance=204.63
    Saved image 8 at C:\Users\Ezekiel\Desktop\microscope1\405nm testing\edith4\image_8.png
    


    
![png](output_0_15.png)
    


    Moving to next position: X=134247, Y=9069
    Position after move: {'x': 134247, 'y': 9069, 'z': 89280}
    Evaluating direction for coarse focus...
    Coarse focus direction: down
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[20], line 159
        156     print(f"Position after move: {microscope.position}")
        158 # Autofocus at current position
    --> 159 best_z, focused_image = autofocus()
        161 # Update Z
        162 starting_pos['z'] = best_z
    

    Cell In[20], line 123, in autofocus()
        122 def autofocus():
    --> 123     coarse_z, _, coarse_image, z_positions, variances = coarse_focus()
        124     fine_z, _, fine_image = fine_focus(coarse_z, z_positions, variances)
        125     return fine_z, fine_image
    

    Cell In[20], line 91, in coarse_focus()
         89 direction = evaluate_focus_direction(coarse_step_size, coarse_num_steps)
         90 print(f"Coarse focus direction: {'up' if direction == 1 else 'down'}")
    ---> 91 best_z, best_var, best_image, z_positions, variances = move_until_drop(coarse_step_size, direction)
         92 print(f"Coarse focus at Z={best_z} with variance={best_var:.2f}")
         93 return best_z, best_var, best_image, z_positions, variances
    

    Cell In[20], line 67, in move_until_drop(step_size, direction)
         65 while True:
         66     move_microscope_rel(direction * step_size)
    ---> 67     image = capture_image()
         68     var = compute_laplacian_variance(image)
         70     positions.append(microscope.position['z'])
    

    Cell In[20], line 36, in capture_image()
         35 def capture_image():
    ---> 36     return np.array(microscope.grab_image())
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\openflexure_microscope_client\microscope_client.py:176, in MicroscopeClient.grab_image(self)
        174 def grab_image(self):
        175     """Grab an image from the stream and return as a PIL image object"""
    --> 176     image = PIL.Image.open(io.BytesIO(self.grab_image_raw()))
        177     return image
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\openflexure_microscope_client\microscope_client.py:120, in MicroscopeClient.grab_image_raw(self)
        119 def grab_image_raw(self):
    --> 120     r = requests.get(self.base_uri + "/streams/snapshot")
        121     r.raise_for_status()
        122     return r.content
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\requests\api.py:73, in get(url, params, **kwargs)
         62 def get(url, params=None, **kwargs):
         63     r"""Sends a GET request.
         64 
         65     :param url: URL for the new :class:`Request` object.
       (...)
         70     :rtype: requests.Response
         71     """
    ---> 73     return request("get", url, params=params, **kwargs)
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\requests\api.py:59, in request(method, url, **kwargs)
         55 # By using the 'with' statement we are sure the session is closed, thus we
         56 # avoid leaving sockets open which can trigger a ResourceWarning in some
         57 # cases, and look like a memory leak in others.
         58 with sessions.Session() as session:
    ---> 59     return session.request(method=method, url=url, **kwargs)
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\requests\sessions.py:589, in Session.request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        584 send_kwargs = {
        585     "timeout": timeout,
        586     "allow_redirects": allow_redirects,
        587 }
        588 send_kwargs.update(settings)
    --> 589 resp = self.send(prep, **send_kwargs)
        591 return resp
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\requests\sessions.py:703, in Session.send(self, request, **kwargs)
        700 start = preferred_clock()
        702 # Send the request
    --> 703 r = adapter.send(request, **kwargs)
        705 # Total elapsed time of the request (approximately)
        706 elapsed = preferred_clock() - start
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\requests\adapters.py:486, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        483     timeout = TimeoutSauce(connect=timeout, read=timeout)
        485 try:
    --> 486     resp = conn.urlopen(
        487         method=request.method,
        488         url=url,
        489         body=request.body,
        490         headers=request.headers,
        491         redirect=False,
        492         assert_same_host=False,
        493         preload_content=False,
        494         decode_content=False,
        495         retries=self.max_retries,
        496         timeout=timeout,
        497         chunked=chunked,
        498     )
        500 except (ProtocolError, OSError) as err:
        501     raise ConnectionError(err, request=request)
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\urllib3\connectionpool.py:791, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, preload_content, decode_content, **response_kw)
        788 response_conn = conn if not release_conn else None
        790 # Make the request on the HTTPConnection object
    --> 791 response = self._make_request(
        792     conn,
        793     method,
        794     url,
        795     timeout=timeout_obj,
        796     body=body,
        797     headers=headers,
        798     chunked=chunked,
        799     retries=retries,
        800     response_conn=response_conn,
        801     preload_content=preload_content,
        802     decode_content=decode_content,
        803     **response_kw,
        804 )
        806 # Everything went great!
        807 clean_exit = True
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\urllib3\connectionpool.py:537, in HTTPConnectionPool._make_request(self, conn, method, url, body, headers, retries, timeout, chunked, response_conn, preload_content, decode_content, enforce_content_length)
        535 # Receive the response from the server
        536 try:
    --> 537     response = conn.getresponse()
        538 except (BaseSSLError, OSError) as e:
        539     self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
    

    File ~\AppData\Local\anaconda3\Lib\site-packages\urllib3\connection.py:461, in HTTPConnection.getresponse(self)
        458 from .response import HTTPResponse
        460 # Get the response from http.client.HTTPConnection
    --> 461 httplib_response = super().getresponse()
        463 try:
        464     assert_header_parsing(httplib_response.msg)
    

    File ~\AppData\Local\anaconda3\Lib\http\client.py:1386, in HTTPConnection.getresponse(self)
       1384 try:
       1385     try:
    -> 1386         response.begin()
       1387     except ConnectionError:
       1388         self.close()
    

    File ~\AppData\Local\anaconda3\Lib\http\client.py:325, in HTTPResponse.begin(self)
        323 # read until we get a non-100 response
        324 while True:
    --> 325     version, status, reason = self._read_status()
        326     if status != CONTINUE:
        327         break
    

    File ~\AppData\Local\anaconda3\Lib\http\client.py:286, in HTTPResponse._read_status(self)
        285 def _read_status(self):
    --> 286     line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        287     if len(line) > _MAXLINE:
        288         raise LineTooLong("status line")
    

    File ~\AppData\Local\anaconda3\Lib\socket.py:706, in SocketIO.readinto(self, b)
        704 while True:
        705     try:
    --> 706         return self._sock.recv_into(b)
        707     except timeout:
        708         self._timeout_occurred = True
    

    KeyboardInterrupt: 



```python

```
