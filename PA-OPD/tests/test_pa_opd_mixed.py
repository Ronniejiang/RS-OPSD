import asyncio
import json

from omegaconf import OmegaConf
from PIL import Image

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.pa_opd_dataset import _REMOVE_HINT, load_pa_opd_direct_jsonl
from verl.utils.dataset.pa_opd_image import load_capped_image
from verl.utils.dataset.rl_dataset import RLHFDataset


def write_source(root, boxed_folder=False):
    for folder in ["images", "teacher_images"] + (["bbox_images"] if boxed_folder else []):
        (root / folder).mkdir(parents=True)
        Image.new("RGB", (100, 50), "red").save(root / folder / "one.png")
    row = {"images": ["images/one.png"], "teacher_images": ["teacher_images/one.png"],
           "problem": f"<image> Which color? {_REMOVE_HINT}\nA. red\nB. blue", "answer": "A"}
    (root / "train.jsonl").write_text(json.dumps(row) + "\n")
    return root / "train.jsonl"


def test_native_bbox_retains_hint_without_bbox_directory(tmp_path):
    file = write_source(tmp_path)
    row = load_pa_opd_direct_jsonl(file, student_image_mode="native_bbox",
          teacher_full_image_mode="native_bbox", separate_teacher_views=True,
          image_max_side=64, source_name="visionopd")[0]
    assert row["data_source"] == "visionopd"
    assert _REMOVE_HINT in row["prompt"][0]["content"]
    assert _REMOVE_HINT in row["teacher_prompt"][0]["content"]
    assert row["teacher_images"][0] == row["images"][0]
    assert row["images"][0]["path"] == str(tmp_path / "images/one.png")


def test_student_rollout_and_teacher_probe_share_resize(tmp_path):
    path = tmp_path / "full.png"
    Image.new("RGB", (100, 50), "red").save(path)
    before = path.read_bytes()
    item = {"type": "image", "image": str(path), "pa_opd_max_side": 64}
    messages = [{"role": "user", "content": [item]}]
    student, _ = asyncio.run(RLHFDataset.process_vision_info(messages, 16, OmegaConf.create({})))
    teacher = RayPPOTrainer._extract_images_from_messages(messages)
    normalized = RayPPOTrainer._normalize_teacher_image({"path": str(path), "pa_opd_max_side": 64})
    assert student[0].size == teacher[0].size == normalized.size == (64, 32)
    assert student[0].tobytes() == teacher[0].tobytes() == normalized.tobytes()
    assert path.read_bytes() == before


def test_small_crop_is_not_upscaled():
    image = Image.new("RGB", (24, 12))
    assert load_capped_image({"image": image, "pa_opd_max_side": 2048}).size == (24, 12)


def test_actual_dataset_concatenates_with_per_source_views(tmp_path):
    geo = write_source(tmp_path / "geo", boxed_folder=True)
    vision = write_source(tmp_path / "vision")
    config = OmegaConf.create({
        "pa_opd_direct_jsonl": True, "pa_opd_student_image_mode": "bbox_images",
        "pa_opd_teacher_full_image_mode": "bbox_images", "pa_opd_separate_teacher_views": True,
        "pa_opd_source_name": "geoevidence", "pa_opd_image_max_side": 2048,
        "filter_overlong_prompts": False, "shuffle": True,
        "pa_opd_source_overrides": [{"path": str(vision), "student_image_mode": "native_bbox",
                                    "teacher_full_image_mode": "native_bbox", "source_name": "visionopd"}],
    })
    data = RLHFDataset([str(geo), str(vision)], tokenizer=None, processor=object(), config=config)
    assert len(data) == 2
    assert [data[i]["data_source"] for i in range(2)] == ["geoevidence", "visionopd"]
    assert data[0]["extra_info"]["student_image_mode"] == "bbox_images"
    assert data[1]["extra_info"]["student_image_mode"] == "native_bbox"
    assert data[0]["extra_info"]["source_jsonl"] != data[1]["extra_info"]["source_jsonl"]
    for i in range(2):
        images = [block for block in data[i]["raw_prompt"][0]["content"] if block["type"] == "image"]
        assert images[0]["pa_opd_max_side"] == 2048
