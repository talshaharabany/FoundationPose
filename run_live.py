from estimater import *
from datareader import *
import argparse
import numpy as np
from transformers import AutoProcessor, Owlv2ForObjectDetection, image_transforms
from torchvision.ops import nms


def generate_bbox_mask(image_shape, bbox):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox.long().tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def convert_bbox(image, bbox):
    img_size = max(image.height, image.width)
    scale_fct = torch.tensor([img_size, img_size, img_size, img_size]).to(bbox.device).unsqueeze(dim=0)
    return bbox * scale_fct


def op_nms(boxes, scores, image_feats, th=0.3):
    keep = nms(boxes, scores, th)
    return boxes[keep], scores[keep], image_feats[keep]


def filter_boxes(boxes, scores, image_feats, th=0.1):
    keep = scores.squeeze() >= th
    return boxes.squeeze()[keep], scores.squeeze()[keep], image_feats.squeeze()[keep]


def get_score(boxes, golden, z, th):
  golden = golden / golden.norm(dim=-1, keepdim=True)
  sim = (z @ golden.T).max(dim=-1)[0]
  if (sim >= th).sum() == 0:
     return None
  boxes = boxes[sim >= th]
  sim = sim[sim >= th]
  return boxes[torch.argmax(sim)]


@torch.no_grad()
def get_feat(image, model, processor, goldens, th_obj=0.1, th_nms=0.3):
  image = Image.fromarray(image)
  inputs = processor(images=image, return_tensors="pt")
  feature_map = model.image_embedder(pixel_values=inputs["pixel_values"].cuda())[0]
  batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
  image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
  boxes = model.box_predictor(image_feats, feature_map)
  boxes = image_transforms.center_to_corners_format(boxes)
  objectness_logits = model.objectness_predictor(image_feats)
  scores = F.sigmoid(objectness_logits)
  boxes, scores, image_feats = filter_boxes(boxes, scores, image_feats, th_obj)
  boxes, scores, image_feats = op_nms(boxes, scores, image_feats, th_nms)
  boxes = convert_bbox(image, boxes)
  query_embeds = model.class_head.dense0(image_feats).unsqueeze(dim=0)
  query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
  return get_score(boxes, goldens, query_embeds.squeeze(), th = 0.7)


def calc_d(wheel_center_pose, hands_center_pose):
  wheel_translation = wheel_center_pose[:3, 3]
  hands_translation = hands_center_pose[:3, 3]
  wheel_rotation = wheel_center_pose[:3, :3]
  hands_translation_in_wheel_frame = np.dot(np.linalg.inv(wheel_rotation), (hands_translation - wheel_translation))
  d = 100 * (np.linalg.norm(hands_translation_in_wheel_frame))
  return int(d)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  data = 'wheel_setup'
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/{data}')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  wheel_mesh = trimesh.load(f'{code_dir}/demo_data/wheel_setup/mesh/textured_simple.obj')

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_wheel_origin, wheel_extents = trimesh.bounds.oriented_bounds(wheel_mesh)
  wheel_bbox = np.stack([-wheel_extents/2, wheel_extents/2], axis=0).reshape(2,3)

  processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
  model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval().half()
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  wheel_est = FoundationPose(model_pts=wheel_mesh.vertices, model_normals=wheel_mesh.vertex_normals, mesh=wheel_mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  
  z_wheel = torch.load('wheel_goldens.pth')
  logging.info("estimator initialization done")
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=480, zfar=np.inf)
  
  root_dir = "/home/talshah/Documents/FoundationPose/output_data/example"
  rgb_dir = os.path.join(root_dir, "rgb")
  depth_dir = os.path.join(root_dir, "depth")
  mask_dir = os.path.join(root_dir, "masks")
  i = 0
  dec = 60000
  video_width, video_height = 853, 480
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video_writer = cv2.VideoWriter('final10.mp4', fourcc, 20, (video_width, video_height))  

  while 1:
    color = cv2.resize()
    depth = cv2.resize()  
    depth = np.nan_to_num(depth, nan=0)
    depth[depth<0.001] = 0
    cv2.imwrite(os.path.join(rgb_dir, f"{i:04d}.png"), color)
    if i % dec == 0:
      wheel_annotation_bbox = get_feat(color, model, processor, z_wheel / z_wheel.norm(dim=-1, keepdim=True))
      if wheel_annotation_bbox is None:
        print('not found')
        time.sleep(1)
        continue
      i = i + 1
      wheel_ob_mask = generate_bbox_mask((480, 853), wheel_annotation_bbox)
      wheel_pose = wheel_est.register(K=reader.K, rgb=color, depth=depth, ob_mask=wheel_ob_mask, iteration=args.est_refine_iter)
    else:
      wheel_pose = wheel_est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
      i = i + 1
    wheel_center_pose = wheel_pose@np.linalg.inv(to_wheel_origin)
    vis = draw_xyz_axis(vis, ob_in_cam=wheel_center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    video_writer.write(vis)
  video_writer.release()
