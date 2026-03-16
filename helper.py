import os
import gc
import cv2
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from functools import partial

from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms

import v2ce



class EventData:
    def __init__(self, events, label, source):
        """
        events : numpy array of event tuples
        label  : string label of the video
        source : original video path
        """
        self.events = events
        self.label = label
        self.source = source

def scale_video(video_frames, target_width = 640, target_height = 480):
    # get original dimensions from frames       
    _, _,  orig_height, orig_width = video_frames.shape

    # Calculate new dimensions for center cropping
    target_aspect = target_width / target_height
    orig_aspect = orig_width / orig_height

    if orig_aspect > target_aspect:
        new_width = int(target_aspect * orig_height)
        new_height = orig_height
    else:
        new_height = int(orig_width / target_aspect)
        new_width = orig_width

    transform = transforms.Compose([
        transforms.CenterCrop((new_height, new_width)),
        transforms.Resize((target_height, target_width), antialias=True ),
    ])

    return transform(video_frames)


def retrieve_videos_by_frame(num_frames, video_paths): # list video with num_frames frames and returns the list
    target_videos_idx = list()

    for i, path in tqdm(enumerate(video_paths), total = len(video_paths), desc='Retriving the indexes of frame specific videos: '):
        decoder = VideoDecoder(path)
        num_frames_in_vid = decoder.metadata.num_frames_from_content
        
        if num_frames_in_vid == num_frames:
            target_videos_idx.append(i)

    return np.array(target_videos_idx)

def load_video_frames(path, scale=True):
    """
    Load video frames from a given path using VideoDecoder.
    Returns frames as a numpy array of shape (T, C, H, W).
    """

    decoder = VideoDecoder(path)
    num_frames = decoder.metadata.num_frames_from_content
    frames = decoder.get_frames_in_range(0, num_frames)
    frames = scale_video(frames, target_width=640, target_height=480) if scale else frames
    return frames

# ---------------------------------------------------------
# Convert RGB → events using V2CE
# ---------------------------------------------------------

def frames_2_events(
        video_indices,
        video_paths,
        labels,
        infer_type='center',
        seq_len=16,
        batch_size=1,
        width=640,
        height=480,
        fps=15,
        stage2_batch_size=24):
    """
    Converts RGB videos to event streams using V2CE.
    Returns a list of EventData objects.
    """

    model = v2ce.get_trained_mode('../V2CE-Toolbox/weights/v2ce_3d.pt')

    outputs = []

    for idx in video_indices:
        label = labels[idx]
        video_path = video_paths[idx]

        video_reader = v2ce.VideoReader(video_path, color_mode='GRAY')

        pred_voxel = v2ce.video_to_voxels(
            model,
            vidcap=video_reader,
            infer_type=infer_type,
            seq_len=seq_len,
            batch_size=batch_size,
            width=width,
            height=height
        )

        L, _, _, H, W = pred_voxel.shape

        stage2_input = pred_voxel.reshape(L, 2, 10, H, W)
        stage2_input = torch.from_numpy(stage2_input).cuda()

        ldati = partial(
            v2ce.sample_voxel_statistical,
            fps=fps,
            bidirectional=False,
            additional_events_strategy='slope'
        )

        # Generate in batches
        batch_events = []
        for i in range(0, stage2_input.shape[0], stage2_batch_size):
            batch_events.extend(ldati(stage2_input[i:i + stage2_batch_size]))

        # Stitch timestamps
        merged = []
        for i, e in enumerate(batch_events):
            e['timestamp'] += int(i * (1 / fps) * 1e6)
            merged.append(e)

        if merged:
            merged = np.concatenate(merged)

            outputs.append(EventData(
                events=merged,
                label=label,
                source=video_path
            ))

        gc.collect()

    return outputs


# ---------------------------------------------------------
# Convert events → visual frames (no saving)
# ---------------------------------------------------------

def events_to_frames(events, fps=15):
    xs = events['x'].astype(int)
    ys = events['y'].astype(int)

    width = xs.max() + 1
    height = ys.max() + 1

    ts = events['timestamp']
    ts = ts - ts.min()

    frame_interval = 1e6 / fps
    frame_ids = (ts / frame_interval).astype(int)
    total_frames = frame_ids.max() + 1

    frames = []

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        mask = (frame_ids == f)
        ev = events[mask]

        if len(ev) > 0:
            xs_f = ev['x'].astype(int)
            ys_f = ev['y'].astype(int)
            ps_f = ev['polarity'].astype(int)

            pos = ps_f == 1   # red
            neg = ps_f == 0   # blue

            frame[ys_f[pos], xs_f[pos]] = (255, 0, 0)
            frame[ys_f[neg], xs_f[neg]] = (0, 0, 255)

        frames.append(frame)

    return frames



# ---------------------------------------------------------
# Optional: save event stream if needed
# ---------------------------------------------------------

def save_event_npz(sample: EventData, out_folder):
    """
    Saves an event stream to NPZ (optional).
    """
    os.makedirs(out_folder, exist_ok=True)

    name = Path(sample.source).stem
    out_path = os.path.join(out_folder, f"{sample.label}_{name}.npz")

    np.savez(out_path, event_stream=sample.events)
    return out_path

def load_event_npz(path):
    data = np.load(path, allow_pickle=True)
    if "event_stream" not in data:
        raise RuntimeError(f"No event_stream found in {path}")
    return data["event_stream"]


def play_video(video_frames,label, fps=15, window_name="Playback"):
    """
    Plays a scaled video continuously at a fixed FPS.
    Input can be:
        (T, H, W, C)  — numpy format
        (T, C, H, W)  — torch tensor format
    """

    # Convert torch tensor → numpy (if needed)
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()

    # Common format: (T, C, H, W) → (T, H, W, C)
    if video_frames.ndim == 4 and video_frames.shape[1] in [1, 3]:
        video_frames = np.transpose(video_frames, (0, 2, 3, 1))  # CHW → HWC

    # Safety check
    if video_frames.ndim != 4:
        raise ValueError("Expected video_frames of shape (T,H,W,C).")

    delay = int(1000 / fps)
    frame_idx = 0
    total_frames = video_frames.shape[0]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = video_frames[frame_idx]

        # Convert RGB → BGR for OpenCV
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f'label: {label}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)        
        cv2.putText(frame, 'Resuloution: {}x{}'.format(frame.shape[1], frame.shape[0]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:
            break

        # Continuous loop
        frame_idx = (frame_idx + 1) % total_frames

    cv2.destroyAllWindows()

def load_paths_labels(view = 'view1_lh_aa', split = 'train'):
    PATH = f'./data/raw/{view}/videos_' + split

    action_map = {
        'a': 'Approach',
        'g': 'Grasp',
        'h': 'Hold',
        'i': 'Insert',
        'm': 'Move',
        'r': 'Rotate',
        's': 'Screw',
        'l': 'Slide',
        'p': 'Place',
        'd': 'Disasemble',
        'n': 'Null/Error',
        'w': 'Withdraw'
    }
    video_paths = sorted(glob(os.path.join(PATH, '*/*.mp4')))
    labels = [action_map[abbr.split('/')[-2][0]] for abbr in video_paths]

    return video_paths, labels
# ----------------------------------------------------------
