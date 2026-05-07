import pyopencl as cl
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import serial
import joblib
from sklearn.svm import SVC

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

class Reservoir:
    def __init__(self, nodes=1, capture_period=0.05, num_frames=50, delay=0, region=[[0, 1], [0, 1]], save_path='models'):
        self.region = region
        self.nodes = nodes
        self.models = [SVC(kernel='linear', probability=True) for _ in range(num_frames)]
        self.feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.x_width, self.y_width = int(self.feed.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.period_between_captures, self.num_frames, self.delay = capture_period, num_frames, delay
        self.combined_model = SVC(kernel='linear', probability=True)  # Combined model
        self.save_path = save_path

        # OpenCL setup
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices(cl.device_type.GPU)[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # OpenCL kernel code for image filtering
        self.kernel_code = """
        __kernel void filter_image(__global const uchar *frame, __global const uchar *background,
                                   __global uchar *result, const float factor) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int index = y * get_global_size(0) + x;

            float diff = frame[index] - background[index];
            result[index] = (uchar)(255 / (1 + exp(-factor * diff)));
        }
        """
        self.program = cl.Program(self.context, self.kernel_code).build()

    def background(self, period=5, con=1):
        frames, start_time = [], time.time()
        while time.time() - start_time < period:
            frames.append(self.capture())
        frame_p = [self.image(frame) for frame in frames]
        background_frame = np.mean(np.array(frame_p), axis=0)
        plt.imshow(background_frame, cmap='gray')
        plt.axis('off')
        plt.show()
        return background_frame

    def capture(self):
        ret, frame = self.feed.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print('Camera Error')

    def image(self, frame, show=False):
        frame = frame[int(self.region[1][0] * self.y_width):int(self.region[1][1] * self.y_width),
                      int(self.region[0][0] * self.x_width):int(self.region[0][1] * self.x_width)]
        if show:
            plt.imshow(frame, cmap='gray')
            plt.axis('off')
            plt.show()
        return frame

    def filter_image(self, frame, background_frame, factor=1, show=False):
        # Ensure frame and background are numpy arrays of type uint8
        frame = frame.astype(np.uint8)
        background_frame = background_frame.astype(np.uint8)
        
        # Prepare result array
        result = np.empty_like(frame)

        # Create OpenCL buffers
        d_frame = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=frame)
        d_background_frame = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=background_frame)
        d_result = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)

        # Execute the OpenCL kernel
        self.program.filter_image(self.queue, frame.shape, None, d_frame, d_background_frame, d_result, np.float32(factor))

        # Copy result back to host
        cl.enqueue_copy(self.queue, result, d_result).wait()

        if show:
            plt.imshow(result, cmap='gray')
            plt.axis('off')
            plt.show()

        return result

    def encode(self, input):
        ser = serial.Serial('COM3', 115200)
        time.sleep(3)
        inp = ''.join(str(value) for value in input) + '\n'
        ser.write(inp.encode())
        ser.close()
        print(inp)


    def reset(self):
        ser = serial.Serial('COM3', 115200)
        time.sleep(4)
        ser.write('000000'.encode())
        ser.close()

    def load(self, inputs):
        Datas = []
        for i in range(len(inputs)):
            X = np.loadtxt(inputs[i], delimiter=',')
            Datas.append(X)
        return Datas

    #def seperate(self, frames, background, show_filtered=False):
        #filtered = [self.filter_image(self.image(frame), background, factor=20, show=show_filtered) for frame in frames]
        #reduced = [Maths(array).AvMatrix([20, 20]).flatten() for array in filtered]
        #return np.array(reduced)
    def seperate(self, frames, background, show_filtered=False, show_original=False):
        if show_original:
            for frame in frames:
                img = self.image(frame)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.show()
    
        filtered = [self.filter_image(self.image(frame), background, factor=0.25) for frame in frames]
        
        if show_filtered:
            for f in filtered:
                plt.imshow(f, cmap='gray')
                plt.axis('off')
                plt.show()
        
        reduced = [Maths(array).AvMatrix([20, 20]).flatten() for array in filtered]
        return np.array(reduced)


    def test(self, samples, Y):
        scores, Yp = np.zeros(len(self.models)), [model.predict(samples[:, n]) for n, model in enumerate(self.models)]
        Yprob = [model.predict_proba(samples[:, n]) for n, model in enumerate(self.models)]
        for n, answerlist in enumerate(Yp):
            for a, answer in enumerate(answerlist):
                if answer == Y[a]:
                    scores[n] += 1
        combined_predictions = self.combined_model.predict(samples.reshape(samples.shape[0], -1))
        combined_accuracy = np.mean(combined_predictions == Y)
        return Yp, Yprob, scores / len(Y), combined_predictions, combined_accuracy

    def train(self, samples, Y):
        for n, model in enumerate(self.models):
            model.fit(samples[:, n], Y)
        combined_samples = samples.reshape(samples.shape[0], -1)  # Combine features
        self.combined_model.fit(combined_samples, Y)
        print('Models have been trained')

    def save_models(self):
        os.makedirs(self.save_path, exist_ok=True)
        for n, model in enumerate(self.models):
            joblib.dump(model, f'{self.save_path}/model_{n}.pkl')
        joblib.dump(self.combined_model, f'{self.save_path}/combined_model.pkl')
        print('Models have been saved')

    def load_models(self):
        for n in range(len(self.models)):
            self.models[n] = joblib.load(f'{self.save_path}/model_{n}.pkl')
        self.combined_model = joblib.load(f'{self.save_path}/combined_model.pkl')
        print('Models have been loaded')

    def release_camera(self):
        """Helper function to explicitly release the camera."""
        if self.feed is not None and self.feed.isOpened():
            self.feed.release()
            #print("Camera feed released.")
        else:
            #print("No camera feed to release.")
            self.feed = None

    def run(self, data, mode, save=False, load=False, analysis=0, show_filtered=False, show_original=False):
        
        self.release_camera()
        self.feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if load:
            self.load_models()
            return  # Exit if loading models

        if data is None:
            return  # If no data provided, return

        inputs, outputs = data
        Samples = []
        background = self.background()
        for i, input in enumerate(inputs):
            input = [int(value) for value in input]
            print(f'Running input: ', input)
            Frames = []
            self.encode(input)
            capture_time = time.time() + self.delay
            while len(Frames) < self.num_frames:
                if time.time() - capture_time > self.period_between_captures:
                    Frames.append(self.capture())
                    capture_time = time.time()
            self.reset()
            print('Input ', i+1, 'complete, resetting reservoir')
            #time.sleep(5) 

            # Call the seperate method with show_filtered and show_original
            Samples.append(self.seperate(Frames, background, show_filtered=show_filtered, show_original=show_original))

        Samples = np.array(Samples)
        if analysis != 0:
            Samples = np.array([[Waveform(model_timeseries).HarmonicCompressor(analysis[0], analysis[1], analysis[2]) for model_timeseries in input_sample] for input_sample in Samples])

        if mode == 'train':
            self.train(Samples, outputs)
            if save:
                self.save_models()
        elif mode == 'test':
            answers, probabilities, results, combined_predictions, combined_accuracy = self.test(Samples, outputs)
            print('predictions: ', combined_predictions)
            return answers, probabilities, results, combined_predictions, combined_accuracy
          # Explicitly release the camera
       

class Maths:
    def __init__(self, matrix):
        self.m = matrix

    def AvMatrix(self, size):
        step_i, step_j = self.m.shape[0] // size[0], self.m.shape[1] // size[1]
        result = np.zeros((size[0], size[1]))
        for i in range(size[0]):
            for j in range(size[1]):
                result[i, j] = np.mean(self.m[i*step_i:i*step_i+step_i, j*step_j:j*step_j+step_j])
        return result

class Waveform:
    def __init__(self, sample):
        self.sample = np.array(sample)

    def HarmonicCompressor(self, CompressionPeriod, HarmonicPeriod, NumHarmonics):
        N = len(self.sample) // CompressionPeriod
        resample = np.array([[self.resonance(self.sample[section*CompressionPeriod:(section+1)*CompressionPeriod], 2*HarmonicPeriod/primes[i]) for i in range(NumHarmonics)] for section in range(N)])
        return resample.flatten()

    def resonance(self, signal, period):
        window = np.hanning(len(signal))
        signal = signal * window
        fft = np.fft.fft(signal)
        fft = np.abs(fft)
        fft = fft[:len(fft)//2]
        return np.mean(fft[int(period/2)-5:int(period/2)+5])
