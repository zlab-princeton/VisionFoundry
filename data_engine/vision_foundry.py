#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import time
import base64
import re
import argparse
import random
import copy
import hashlib
import traceback
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from collections import Counter
from pathlib import Path
import threading

import requests
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================== Config & Constants ==========================
@dataclass
class ProviderConfig:
    type: str
    model: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIConfig:
    providers: Dict[str, ProviderConfig]

@dataclass
class ProviderRuntime:
    type: str
    model: str
    client: Any
    base_url: Optional[str] = None
    timeout: Optional[int] = None

DEFAULT_API_CONFIG = {
    "providers": {
        "text": {
            "type": "openai",
            "model": "gpt-5.2",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1"
        },
        "embed": {
            "type": "openai",
            "model": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1"
        },
        "image": {
            "type": "gemini",
            "model": "gemini-2.5-flash-image",
            "api_key_env": "GEMINI_API_KEY"
        },
        "check": {
            "type": "gemini",
            "model": "gemini-3.0-pro",
            "api_key_env": "GEMINI_API_KEY"
        }
    }
}

TIMEOUT = 60
RETRY_SLEEP = 8
NUM_SECONDS_TO_SLEEP = 10
MAX_ITEMS_PER_LLM_CALL = 50
MODEL_TEXT = ""
MODEL_IMAGE = ""
MODEL_CHECK = ""

def _resolve_api_key(pcfg: ProviderConfig) -> Optional[str]:
    if pcfg.api_key:
        return pcfg.api_key
    if pcfg.api_key_env:
        return os.environ.get(pcfg.api_key_env)
    return None

def load_api_config(path: Optional[str]) -> APIConfig:
    cfg = DEFAULT_API_CONFIG
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    providers = {}
    for name, raw in cfg.get("providers", {}).items():
        providers[name] = ProviderConfig(
            type=raw.get("type", "openai"),
            model=raw.get("model", ""),
            api_key=raw.get("api_key"),
            api_key_env=raw.get("api_key_env"),
            base_url=raw.get("base_url"),
            timeout=raw.get("timeout"),
            extra=raw.get("extra", {})
        )
    return APIConfig(providers=providers)

def build_providers(cfg: APIConfig) -> Dict[str, ProviderRuntime]:
    providers = {}
    for name, pcfg in cfg.providers.items():
        api_key = _resolve_api_key(pcfg)
        if pcfg.type == "openai":
            client = OpenAI(api_key=api_key, base_url=pcfg.base_url)
        elif pcfg.type == "gemini":
            try:
                from google import genai
            except Exception as e:
                raise RuntimeError(
                    "google-genai is required for Gemini providers. "
                    "Install with: pip install google-genai"
                ) from e
            client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider type: {pcfg.type}")
        providers[name] = ProviderRuntime(
            type=pcfg.type,
            model=pcfg.model,
            client=client,
            base_url=pcfg.base_url,
            timeout=pcfg.timeout
        )
    return providers

# ========================== Data Models ==========================
@dataclass
class PoolConfig:
    
    objects: Optional[List[str]] = None
    attributes: Optional[List[str]] = None
    scenes: Optional[List[str]] = None
    styles: Optional[List[str]] = None
    custom_attributes: Dict[str, List[str]] = field(default_factory=dict)
    generate_missing: bool = True
    use_global_pool: Optional[str] = None

@dataclass
class TaskConfig:
    
    task_id: str
    description: str
    mode: str = "single"
    num_objects: int = 1
    num_images: int = 1
    multi_image_form: str = "story_chain"
    constraints: List[str] = field(default_factory=list)
    generation_style: str = "balanced"
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy"])
    pool_size: int = 20
    
    pool_config: Optional[PoolConfig] = None

@dataclass
class Entity:
    
    object: Union[str, List[str]]
    attribute: Optional[str] = None
    scene: Optional[str] = None
    style: Optional[str] = None
    custom: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ImagePromptSpec:
    
    index: int
    type: str
    prompt: str
    ref_image_index: Optional[int] = None

@dataclass
class MultiImageSpec:
    
    image_prompts: List[ImagePromptSpec]
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "image_prompts": [
                {
                    "index": ip.index,
                    "type": ip.type,
                    "prompt": ip.prompt,
                    "ref_image_index": ip.ref_image_index
                }
                for ip in self.image_prompts
            ],
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata
        }

def load_config_from_json(config_path: str) -> TaskConfig:
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = json.load(f)
    
    pool_dict = cfg_dict.get("pool_config", {})
    pool_config = PoolConfig(
        objects=pool_dict.get("objects"),
        attributes=pool_dict.get("attributes"),
        scenes=pool_dict.get("scenes"),
        styles=pool_dict.get("styles"),
        custom_attributes=pool_dict.get("custom_attributes", {}),
        generate_missing=pool_dict.get("generate_missing", True),
        use_global_pool=pool_dict.get("use_global_pool")
    )
    
    task_config = TaskConfig(
        task_id=cfg_dict.get("task_id", "default"),
        description=cfg_dict.get("description", ""),
        mode=cfg_dict.get("mode", "single"),
        num_objects=cfg_dict.get("num_objects", 1),
        num_images=cfg_dict.get("num_images", 1),
        multi_image_form=cfg_dict.get("multi_image_form", "story_chain"),
        constraints=cfg_dict.get("constraints", []),
        generation_style=cfg_dict.get("generation_style", "balanced"),
        difficulty_levels=cfg_dict.get("difficulty_levels", ["easy"]),
        pool_config=pool_config
    )
    return task_config

def save_config_template(output_path: str):
    
    template = {
        "task_id": "spatial_understanding",
        "description": "Understanding spatial relationships between objects",
        "mode": "single",
        "num_objects": 1,
        "num_images": 1,
        "multi_image_form": "story_chain",
        "constraints": [
            "must involve spatial relations",
            "clear visual composition"
        ],
        "generation_style": "balanced",
        "difficulty_levels": ["easy", "medium"],
        "pool_config": {
            "objects": ["red car", "blue building", "green tree"],
            "attributes": None,
            "scenes": ["urban street", "countryside"],
            "styles": ["photorealistic"],
            "custom_attributes": {
                "spatial_relations": ["left", "right", "above", "below", "in front of", "behind"],
                "depth_levels": ["close", "far", "medium distance"]
            },
            "generate_missing": True,
            "use_global_pool": None
        }
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"Config template saved to: {output_path}")
    
    template_multi = {
        "task_id": "narrative_reasoning",
        "description": "Understanding narrative progression and object changes across images",
        "mode": "multi",
        "num_objects": 2,
        "num_images": 3,
        "multi_image_form": "story_chain",
        "constraints": [
            "must show clear narrative progression",
            "object transformations must be visible"
        ],
        "generation_style": "balanced",
        "difficulty_levels": ["easy"],
        "pool_config": {
            "objects": ["car", "person", "building"],
            "attributes": None,
            "scenes": ["street", "park"],
            "styles": ["photorealistic"],
            "custom_attributes": {},
            "generate_missing": True,
            "use_global_pool": None
        }
    }
    
    template_multi_path = output_path.replace(".json", "_multi.json")
    with open(template_multi_path, 'w', encoding='utf-8') as f:
        json.dump(template_multi, f, ensure_ascii=False, indent=2)
    print(f"Config template (multi-image) saved to: {template_multi_path}")

def save_jsonl(lines: List[dict], path: str, lock=None):
    
    mode = 'a' if os.path.exists(path) else 'w'
    if lock:
        with lock:
            with open(path, mode, encoding='utf-8') as f:
                for obj in lines:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    else:
        with open(path, mode, encoding='utf-8') as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def extract_image_url(content: str) -> Optional[str]:
    
    if not content:
        return None
    # 1) data URI
    m = re.search(r'(data:image/(png|jpeg|jpg);base64,[A-Za-z0-9+/=\n\r]+)', content)
    if m:
        return m.group(1)
    # 2) markdown image
    m = re.search(r'!\[[^\]]*\]\(\s*(https?://[^\s)]+)\s*\)', content)
    if m:
        return m.group(1)
    m = re.search(r'\(\s*(https?://[^\s)]+)\s*\)', content)
    if m:
        return m.group(1)
    m = re.search(r'(https?://[^\s"\'<>]+\.(png|jpg|jpeg|webp|gif|bmp)(?:\?[^\s]*)?)', content)
    if m:
        return m.group(1)
    m = re.search(r'(https?://[^\s"\'<>]+)', content)
    if m:
        return m.group(1)
    return None

def download_bytes(url: str) -> bytes:
    
    if url.startswith('data:image/'):
        return base64.b64decode(url.split(',', 1)[1])
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

def save_image(image_bytes: bytes, output_path: str):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_bytes)

def _check_api_error_and_handle(resp):
    try:
        raw = resp.model_dump()
    except Exception:
        try:
            raw = resp.to_dict()
        except Exception:
            raw = None
    if isinstance(raw, dict) and raw.get('error'):
        err = raw['error']
        code = err.get('code', '')
        msg = err.get('message', str(err))
        if code == 'channel_rpm_limit_exceeded':
            print(f"[rate limit] API RPM limit: {code}. Sleeping {NUM_SECONDS_TO_SLEEP}s")
            time.sleep(NUM_SECONDS_TO_SLEEP)
            return 'retry'
        raise RuntimeError(f"API error: {code} - {msg}")
    return raw

def _data_uri_from_bytes(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"

def _bytes_from_url_or_data(image_url: str) -> bytes:
    if image_url.startswith('data:image/'):
        return base64.b64decode(image_url.split(',', 1)[1])
    return download_bytes(image_url)

def _extract_gemini_image(resp) -> Optional[Tuple[bytes, str]]:
    candidates = getattr(resp, 'candidates', None) or []
    for cand in candidates:
        content = getattr(cand, 'content', None)
        parts = getattr(content, 'parts', None) or []
        for part in parts:
            inline = getattr(part, 'inline_data', None)
            if inline is None:
                continue
            data = getattr(inline, 'data', None)
            mime = getattr(inline, 'mime_type', None) or 'image/png'
            if data is None:
                continue
            if isinstance(data, str):
                try:
                    return base64.b64decode(data), mime
                except Exception:
                    return data.encode('utf-8'), mime
            return data, mime
    return None

def generate_image(prompt: str, provider: ProviderRuntime, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            if provider.type == 'openai':
                resp = provider.client.images.generate(
                    model=provider.model,
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                if not resp.data:
                    raise ValueError('No image returned')
                item = resp.data[0]
                b64 = getattr(item, 'b64_json', None)
                if b64:
                    return f"data:image/png;base64,{b64}"
                url = getattr(item, 'url', None)
                if url:
                    return url
                raise ValueError('Unknown image response format')
            elif provider.type == 'gemini':
                from PIL import Image
                response = provider.client.models.generate_content(
                    model=provider.model,
                    contents=prompt
                )
                image = _extract_gemini_image(response)
                if not image:
                    raise ValueError('No image in Gemini response')
                image_bytes, mime = image
                return _data_uri_from_bytes(image_bytes, mime)
            else:
                raise ValueError(f"Unsupported provider: {provider.type}")
        except Exception as e:
            print(f"[generate_image] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(RETRY_SLEEP)

def edit_image_get_url(input_bytes: bytes, edit_prompt: str, provider: ProviderRuntime, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            if provider.type == 'openai':
                import io
                image_file = io.BytesIO(input_bytes)
                image_file.name = 'input.png'
                resp = provider.client.images.edit(
                    model=provider.model,
                    image=image_file,
                    prompt=edit_prompt,
                    n=1,
                    size="1024x1024"
                )
                if not resp.data:
                    raise ValueError('No image returned')
                item = resp.data[0]
                b64 = getattr(item, 'b64_json', None)
                if b64:
                    return f"data:image/png;base64,{b64}"
                url = getattr(item, 'url', None)
                if url:
                    return url
                raise ValueError('Unknown image response format')
            elif provider.type == 'gemini':
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(input_bytes))
                response = provider.client.models.generate_content(
                    model=provider.model,
                    contents=[edit_prompt, image]
                )
                image = _extract_gemini_image(response)
                if not image:
                    raise ValueError('No image in Gemini response')
                image_bytes, mime = image
                return _data_uri_from_bytes(image_bytes, mime)
            else:
                raise ValueError(f"Unsupported provider: {provider.type}")
        except Exception as e:
            print(f"[edit_image_get_url] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(RETRY_SLEEP)

def check_image_consistency(statement: str, image_url: str, provider: ProviderRuntime, max_retries: int = 3) -> bool:
    try:
        image_bytes = _bytes_from_url_or_data(image_url)

        system_prompt = (
            "You are an image verification assistant. Decide if an image matches the statement overall, "
            "ignoring minor details. ANSWER WITH EXACTLY ONE LINE: 'Answer: YES' or 'Answer: NO'"
        )
        user_text = f"Statement: '{statement}'\n\nDoes this image match?"

        def parse_answer(text: str) -> Optional[str]:
            matches = re.findall(r'Answer\s*:\s*(YES|NO)', text, flags=re.IGNORECASE)
            if matches:
                return matches[-1].lower()
            return None

        for attempt in range(1, max_retries + 1):
            try:
                if provider.type == 'openai':
                    b64_image = _data_uri_from_bytes(image_bytes)
                    user_content = [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": b64_image}}
                    ]
                    response = provider.client.chat.completions.create(
                        model=provider.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        max_tokens=4000,
                        temperature=0.0
                    )
                    content = response.choices[0].message.content.strip()
                elif provider.type == 'gemini':
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_bytes))
                    prompt = f"{system_prompt}\n\n{user_text}"
                    response = provider.client.models.generate_content(
                        model=provider.model,
                        contents=[prompt, image]
                    )
                    content = (getattr(response, 'text', '') or '').strip()
                else:
                    raise ValueError(f"Unsupported provider: {provider.type}")

                answer = parse_answer(content)
                return answer == 'yes'
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(RETRY_SLEEP)
    except Exception as e:
        print(f"check_image_consistency error: {e}")
        return False

def verify_multi_image_consistency(statement: str, image_urls: List[str], provider: ProviderRuntime, max_retries: int = 3) -> bool:
    try:
        image_bytes_list = []
        for url in image_urls:
            image_bytes_list.append(_bytes_from_url_or_data(url))

        system_prompt = (
            "You are a multi-image verification assistant. "
            "Examine ALL images together as a unified visual context. "
            "Decide if the images collectively support the given statement. "
            "ANSWER WITH EXACTLY ONE LINE: 'Answer: YES' or 'Answer: NO'"
        )
        user_text = f"Statement to verify: '{statement}'\\n\\nDo these images collectively support this statement?"

        def parse_answer(text: str) -> Optional[str]:
            matches = re.findall(r'Answer\\s*:\\s*(YES|NO)', text, flags=re.IGNORECASE)
            if matches:
                return matches[-1].lower()
            return None

        for attempt in range(1, max_retries + 1):
            try:
                if provider.type == 'openai':
                    image_contents = []
                    for i, image_bytes in enumerate(image_bytes_list):
                        image_contents.append({"type": "text", "text": f"[Image {i}]"})
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": _data_uri_from_bytes(image_bytes)}
                        })
                    image_contents.append({"type": "text", "text": user_text})
                    response = provider.client.chat.completions.create(
                        model=provider.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": image_contents}
                        ],
                        max_tokens=4000,
                        temperature=0.0
                    )
                    content = response.choices[0].message.content.strip()
                elif provider.type == 'gemini':
                    from PIL import Image
                    import io
                    parts = [f"{system_prompt}\\n\\n{user_text}"]
                    for image_bytes in image_bytes_list:
                        parts.append(Image.open(io.BytesIO(image_bytes)))
                    response = provider.client.models.generate_content(
                        model=provider.model,
                        contents=parts
                    )
                    content = (getattr(response, 'text', '') or '').strip()
                else:
                    raise ValueError(f"Unsupported provider: {provider.type}")

                answer = parse_answer(content)
                return answer == 'yes'
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(RETRY_SLEEP)
    except Exception as e:
        print(f"verify_multi_image_consistency error: {e}")
        return False

def generate_statement_from_qa(question: str, answer: str, client: OpenAI) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[
                {"role": "system", "content": "Rewrite question+answer into ONE concise sentence. Output only the sentence."},
                {"role": "user", "content": f"Q: {question}\nA: {answer}"}
            ],
            temperature=0.0,
            max_tokens=200
        )
        s = resp.choices[0].message.content.strip().split('\n')[0]
        if not s.endswith('.'):
            s += '.'
        return s
    except Exception as e:
        print(f"Statement gen failed: {e}")
        return f"The answer to '{question}' is {answer}."

def get_embeddings(texts: List[str], client: OpenAI, model: str = "text-embedding-3-small") -> List[List[float]]:
    
    if not texts:
        return []
    
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

def deduplicate_with_embedding(items: List[str], client: OpenAI, threshold: float = 0.95) -> List[str]:
    
    if not items:
        return []
    
    if len(items) <= 1:
        return items
    
    embeddings = get_embeddings(items, client)
    if not embeddings or len(embeddings) != len(items):
        print(f"Warning: Failed to get embeddings for all items. Skipping deduplication.")
        return items
    
    unique_indices = [0]
    embeddings = np.array(embeddings)
    
    for i in range(1, len(embeddings)):
        is_duplicate = False
        for j in unique_indices:
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
            )
            if similarity > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_indices.append(i)
    
    unique_items = [items[i] for i in unique_indices]
    print(f"  Deduplication: {len(items)} -> {len(unique_items)} items (removed {len(items) - len(unique_items)})")
    return unique_items

def build_adaptive_pool(task_config: TaskConfig, client: OpenAI, objects_size: int = 20, attributes_size: int = 20, scenes_size: int = 20, styles_size: int = 20, max_items_per_call: int = MAX_ITEMS_PER_LLM_CALL, llm_decide_attr_size: bool = False) -> PoolConfig:
    
    pool_cfg = task_config.pool_config
    if pool_cfg is None:
        pool_cfg = PoolConfig()
    
    if pool_cfg.use_global_pool and os.path.exists(pool_cfg.use_global_pool):
        print(f"Loading global pool from: {pool_cfg.use_global_pool}")
        with open(pool_cfg.use_global_pool, 'r', encoding='utf-8') as f:
            global_pool = json.load(f)
        if pool_cfg.objects is None:
            pool_cfg.objects = global_pool.get("objects", [])
        if pool_cfg.attributes is None:
            pool_cfg.attributes = global_pool.get("attributes", [])
        if pool_cfg.scenes is None:
            pool_cfg.scenes = global_pool.get("scenes", [])
        if pool_cfg.styles is None:
            pool_cfg.styles = global_pool.get("styles", [])
    
    if llm_decide_attr_size and pool_cfg.attributes is None:
        print("Using LLM to estimate optimal attribute pool size...")
        attributes_size = estimate_attribute_size_with_llm(task_config, client)
    
    if pool_cfg.generate_missing:
        categories_to_generate = []
        if pool_cfg.objects is None:
            categories_to_generate.append(("objects", "main objects", objects_size))
        if pool_cfg.attributes is None:
            categories_to_generate.append(("attributes", "visual attributes", attributes_size))
        if pool_cfg.scenes is None:
            categories_to_generate.append(("scenes", "background scenes", scenes_size))
        if pool_cfg.styles is None:
            categories_to_generate.append(("styles", "artistic/realistic styles", styles_size))
        
        if categories_to_generate:
            print(f"Generating missing pool lists for: {[c[0] for c in categories_to_generate]}")
            print(f"  objects_size={objects_size}, attributes_size={attributes_size}, scenes_size={scenes_size}, styles_size={styles_size}")
            print(f"  max_items_per_call={max_items_per_call}")
            _generate_pool_lists_batch(task_config, categories_to_generate, pool_cfg, client, max_items_per_call)
    
    for custom_key, custom_list in pool_cfg.custom_attributes.items():
        if custom_list is None and pool_cfg.generate_missing:
            print(f"Generating custom attribute: {custom_key}")
            pool_cfg.custom_attributes[custom_key] = _generate_custom_attribute(
                task_config, custom_key, client
            )
    
    print(f"Pool ready: objects={len(pool_cfg.objects or [])} | attributes={len(pool_cfg.attributes or [])} "
          f"| scenes={len(pool_cfg.scenes or [])} | styles={len(pool_cfg.styles or [])} "
          f"| custom={list(pool_cfg.custom_attributes.keys())}")
    
    return pool_cfg

def _generate_pool_lists_batch(task_config: TaskConfig, categories: List[Tuple], pool_cfg: PoolConfig, client: OpenAI, max_items_per_call: int = MAX_ITEMS_PER_LLM_CALL):
 
    for key, desc, target_count in categories:
        print(f"  Generating {key} (target: {target_count} items, max per call: {max_items_per_call})")
        all_items = []
        
        batches_generated = 0
        while len(all_items) < target_count:
            remaining = target_count - len(all_items)
            items_to_generate = min(max_items_per_call, remaining)
            batches_generated += 1
            
            print(f"    Batch {batches_generated}: generating {items_to_generate} items...")
            
            prompt = f"Generate {items_to_generate} diverse {desc} suitable for the task: '{task_config.description}'. " \
                    f"Return ONLY a valid JSON array of strings, no markdown, no explanation."
            
            for retry in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_TEXT,
                        messages=[
                            {"role": "system", "content": "You are a creative assistant. Output ONLY valid JSON array of strings."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        max_tokens=8000
                    )
                    txt = resp.choices[0].message.content.strip()
                    txt = txt.replace("```json", "").replace("```", "").strip()
                    batch_items = json.loads(txt)
                    
                    if isinstance(batch_items, list) and len(batch_items) > 0:
                        batch_items = [str(x).strip() for x in batch_items if str(x).strip()]
                        print(f"      Generated {len(batch_items)} items")
                        
                        if all_items:
                            combined = all_items + batch_items
                            print(f"      Before dedup: {len(combined)} items")
                            combined = deduplicate_with_embedding(combined, client, threshold=0.93)
                            print(f"      After dedup: {len(combined)} items")
                            all_items = combined
                        else:
                            all_items.extend(batch_items)
                        
                        print(f"      Total collected: {len(all_items)}/{target_count}")
                        break
                    else:
                        raise ValueError(f"Invalid response format")
                
                except Exception as e:
                    print(f"      Failed (retry {retry}): {e}")
                    if retry < 2:
                        time.sleep(RETRY_SLEEP)
                    else:
                        print(f"      Exhausted retries for batch {batches_generated}")
        
        final_items = deduplicate_with_embedding(all_items, client, threshold=0.93)
        final_items = final_items[:target_count]
        
        if key == "objects":
            pool_cfg.objects = final_items
        elif key == "attributes":
            pool_cfg.attributes = final_items
        elif key == "scenes":
            pool_cfg.scenes = final_items
        elif key == "styles":
            pool_cfg.styles = final_items
        
        print(f"  Final {key}: {len(final_items)} items")

def estimate_attribute_size_with_llm(task_config: TaskConfig, client: OpenAI) -> int:
    prompt = (
        f"For the task '{task_config.description}', estimate how many distinct visual "
        "attributes are needed to cover the task. Respond with ONLY a single integer "
        "between 10 and 50."
    )
    
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": "You are a data curation expert. Respond with ONLY a single integer number, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            txt = resp.choices[0].message.content.strip()
            numbers = re.findall(r'\d+', txt)
            if numbers:
                estimated_size = int(numbers[0])
                estimated_size = max(10, min(50, estimated_size))
                print(f"  LLM estimated attribute size: {estimated_size}")
                return estimated_size
        except Exception as e:
            print(f"Failed to estimate attribute size with LLM: {e}")
            if retry < 2:
                time.sleep(RETRY_SLEEP)
    
    print("  Failed to get LLM estimate, using default: 20")
    return 20

def _generate_custom_attribute(task_config: TaskConfig, attr_name: str, client: OpenAI, count: int = 15) -> List[str]:
    
    prompt = f"""For the task '{task_config.description}', generate {count} diverse options for '{attr_name}'.
These should be specific, varied, and visually verifiable if possible.
Return ONLY a valid JSON array of strings."""
    
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": "Output ONLY valid JSON array of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=4000
            )
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            lst = json.loads(txt)
            if isinstance(lst, list) and len(lst) > 0:
                return [str(x).strip() for x in lst if str(x).strip()]
        except Exception as e:
            print(f"Failed to generate {attr_name}: {e}")
            if retry < 2:
                time.sleep(RETRY_SLEEP)
    
    return []

class PoolSampler:
    

    def __init__(self, combination_pool: List[Dict], pool_cfg: PoolConfig):
        self.pool = combination_pool
        self.available_indices = list(range(len(combination_pool)))
        self.current_pos = 0

        self.objects = pool_cfg.objects
        self.obj2idx = {o: i for i, o in enumerate(self.objects)}
        self.object_usage = [0] * len(self.objects)

    def _sample_object_min_reuse(self) -> str:
        min_u = min(self.object_usage)
        candidates = [i for i, u in enumerate(self.object_usage) if u == min_u]
        idx = random.choice(candidates)
        self.object_usage[idx] += 1
        return self.objects[idx]

    def sample(self, task_config: TaskConfig) -> Optional[Entity]:
        if self.current_pos >= len(self.available_indices):
            return None

        idx = self.available_indices[self.current_pos]
        self.current_pos += 1

        sel_combo = self.pool[idx]

        obj = sel_combo.get('object', '')

        if task_config.num_objects == 1 or not isinstance(obj, str):
            objects = obj

        else:
            objects_list = []
            used = set()

            objects_list.append(obj)
            used.add(obj)

            anchor_idx = self.obj2idx.get(obj)
            if anchor_idx is not None:
                self.object_usage[anchor_idx] += 1

            while len(objects_list) < task_config.num_objects:
                o = self._sample_object_min_reuse()
                if o not in used:
                    objects_list.append(o)
                    used.add(o)

            objects = objects_list

        return Entity(
            object=objects,
            attribute=sel_combo.get('attribute'),
            scene=sel_combo.get('scene'),
            style=sel_combo.get('style'),
            custom={k: v for k, v in sel_combo.items()
                    if k not in ['object', 'attribute', 'scene', 'style']}
        )


def sample_entity(combination_pool: List[Dict], task_config: TaskConfig, usage_counts: Dict = None) -> Entity:
    
    if not combination_pool:
        return None
    
    idx = random.randint(0, len(combination_pool) - 1)
    sel_combo = combination_pool[idx]
    
    objects = sel_combo.get('object', '')
    if task_config.num_objects > 1 and isinstance(objects, str):
        objects_list = [objects]
        for _ in range(task_config.num_objects - 1):
            if combination_pool:
                other_idx = random.randint(0, len(combination_pool) - 1)
                other_combo = combination_pool[other_idx]
                objects_list.append(other_combo.get('object', ''))
        objects = objects_list
    
    entity = Entity(
        object=objects,
        attribute=sel_combo.get('attribute'),
        scene=sel_combo.get('scene'),
        style=sel_combo.get('style'),
        custom={k: v for k, v in sel_combo.items() if k not in ['object', 'attribute', 'scene', 'style']}
    )
    return entity

def build_combination_pool(pool_cfg: PoolConfig, task_config: TaskConfig) -> List[Dict]:
    
    import itertools
    
    dims = {}
    if pool_cfg.objects:
        dims['object'] = pool_cfg.objects
    if pool_cfg.attributes:
        dims['attribute'] = pool_cfg.attributes
    if pool_cfg.scenes:
        dims['scene'] = pool_cfg.scenes
    if pool_cfg.styles:
        dims['style'] = pool_cfg.styles
    
    for key, lst in pool_cfg.custom_attributes.items():
        if lst:
            dims[key] = lst
    
    if not dims:
        print("Warning: No pool dimensions found!")
        return []
    
    keys = list(dims.keys())
    values = list(dims.values())
    combinations = []
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        combinations.append(combo_dict)
    
    random.shuffle(combinations)
    print(f"Built combination pool: {len(combinations)} total combinations from {len(dims)} dimensions")
    return combinations

def build_generation_system_prompt(task_config: TaskConfig) -> str:
    
    base = f"""You are an expert VQA data creator for the task: "{task_config.task_id}"
Task description: {task_config.description}
"""
    
    if task_config.num_objects > 1:
        base += f"This task requires {task_config.num_objects} objects in the image with clear relationships.\n"
    
    if task_config.mode == "multi":
        base += f"MULTI-IMAGE MODE: Generate {task_config.num_images} interconnected images with a coherent story.\n"
    
    if "spatial" in task_config.task_id.lower() or any("spatial" in c.lower() for c in task_config.constraints):
        base += "SPATIAL FOCUS: The question MUST involve spatial relationships between objects.\n"
    
    if "color" in task_config.task_id.lower() or any("color" in c.lower() for c in task_config.constraints):
        base += "COLOR FOCUS: Object colors must be distinct, accurate, and central to the question.\n"
    
    base += f"""
CRITICAL RULES:
1. The answer-determining fact MUST be 100% visually verifiable from the final image.
2. Text prompt must explicitly describe content matching the correct answer.
3. Never rely on invisible properties.
4. Generate deterministic, unambiguous questions.

Constraints: {', '.join(task_config.constraints) if task_config.constraints else 'None'}

Return EXACTLY ONE JSON object with keys:
- "prompt": extremely detailed text-to-image prompt (English)
- "question": clear VQA question
- "answer": clear deterministic answer
- "metadata": {{"difficulty": "easy", "category": "{task_config.task_id}", "num_objects": {task_config.num_objects}}}
"""
    return base

def build_multi_image_system_prompt(task_config: TaskConfig) -> str:
    
    prompt = f"""You are an expert multi-image VQA data creator for task: "{task_config.task_id}"
Task description: {task_config.description}

MULTI-IMAGE CONTEXT:
- Number of images: {task_config.num_images}
- Multi-image form: {task_config.multi_image_form}
- You will design a visual narrative spanning {task_config.num_images} images
- ONE cross-image question that requires viewing all images
- ONE deterministic answer based on visual evidence across all images

MULTI-IMAGE FORM DEFINITIONS:
"""
    
    if task_config.multi_image_form == "multi_generate":
        prompt += f"""
  "multi_generate": All images are generated independently (no edits).
  - Each image is generated from scratch
  - The narrative must still be coherent across images
  - Question requires comparing multiple generated images
"""
    
    elif task_config.multi_image_form == "story_chain":
        prompt += f"""
  "story_chain": Image sequence forms a narrative progression.
  - Image 0: GENERATE from scratch (initial scene)
  - Image 1..{task_config.num_images-1}: EDIT from previous image (continuation/modification)
  - Question asks about narrative: "What changed?", "What happened next?", "Why is...different?"
  - Each edit must preserve previous content but add/modify for story progression
"""
    
    elif task_config.multi_image_form == "mixed":
        prompt += f"""
  "mixed": Mix of generation and editing.
  - Some images generated, some edited
  - Flexible narrative structure
  - You decide which images to generate vs edit based on story logic
"""
    
    prompt += f"""

ENTITY ROLE:
The entity (object, scene, style) provides semantic anchor across all images.
Use it to ensure visual consistency while the narrative progresses.

CONSTRAINTS:
{', '.join(task_config.constraints) if task_config.constraints else 'None'}

OUTPUT REQUIREMENTS:
Return EXACTLY ONE JSON object with structure:
{{
  "image_prompts": [
    {{
      "index": 0,
      "type": "generate" or "edit",
      "prompt": "<detailed text-to-image or editing prompt>",
      "ref_image_index": null or <if type="edit", index of source image>
    }},
    ... ({task_config.num_images} total entries)
  ],
  "question": "<ONE cross-image question>",
  "answer": "<ONE clear deterministic answer>",
  "metadata": {{
    "difficulty": "easy",
    "category": "{task_config.task_id}",
    "num_images": {task_config.num_images},
    "multi_image_form": "{task_config.multi_image_form}",
    "narrative_summary": "<brief description of story>"
  }}
}}

CRITICAL RULES FOR MULTI-IMAGE:
1. question MUST be answerable only by viewing ALL {task_config.num_images} images together
2. answer MUST be 100% visually verifiable from the image sequence
3. For "story_chain": each edit must reference previous image and show clear progression
4. Each image_prompt must be explicit and deterministic (no ambiguity)
5. Do NOT rely on external knowledge - everything must be visible in images
6. Ensure the entity (object/scene/style) appears consistently across all images

VALIDATION:
- All {task_config.num_images} image_prompts present? YES
- question requires viewing all images? YES
- answer is deterministic? YES
- narrative is coherent? YES
"""
    
    return prompt

def generate_one_case(task_config: TaskConfig, entity: Entity, client: OpenAI) -> Optional[Dict]:
    
    entity_json = json.dumps(entity.to_dict(), ensure_ascii=False)
    system_prompt = build_generation_system_prompt(task_config)
    
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Entity:\n{entity_json}\n\nGenerate ONE case."}
                ],
                temperature=0.0,
                max_tokens=4000
            )
            check = _check_api_error_and_handle(resp)
            if check == 'retry':
                continue
            
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            data = json.loads(txt)
            
            if all(k in data for k in ["prompt", "question", "answer"]):
                return data
            else:
                print(f"[generate_one_case] Missing keys (attempt {attempt})")
        except Exception as e:
            print(f"[generate_one_case] Attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            time.sleep(RETRY_SLEEP)
    
    return None

def generate_multi_image_spec(task_config: TaskConfig, entity: Entity, client: OpenAI) -> Optional[MultiImageSpec]:
    
    entity_json = json.dumps(entity.to_dict(), ensure_ascii=False)
    system_prompt = build_multi_image_system_prompt(task_config)
    
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Entity:\n{entity_json}\n\nGenerate ONE multi-image VQA case."}
                ],
                temperature=0.0,
                max_tokens=8000
            )
            check = _check_api_error_and_handle(resp)
            if check == 'retry':
                continue
            
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            spec_dict = json.loads(txt)
            
            if not all(k in spec_dict for k in ["image_prompts", "question", "answer"]):
                print(f"[generate_multi_image_spec] Missing top-level keys (attempt {attempt})")
                continue
            
            image_prompts = []
            for ip_dict in spec_dict.get("image_prompts", []):
                try:
                    ip = ImagePromptSpec(
                        index=ip_dict.get("index"),
                        type=ip_dict.get("type"),
                        prompt=ip_dict.get("prompt"),
                        ref_image_index=ip_dict.get("ref_image_index")
                    )
                    image_prompts.append(ip)
                except Exception as e:
                    print(f"[generate_multi_image_spec] Failed to parse image_prompt: {e}")
                    continue
            
            if len(image_prompts) != task_config.num_images:
                print(f"[generate_multi_image_spec] Expected {task_config.num_images} images, got {len(image_prompts)}")
                continue
            
            spec = MultiImageSpec(
                image_prompts=image_prompts,
                question=spec_dict.get("question", ""),
                answer=spec_dict.get("answer", ""),
                metadata=spec_dict.get("metadata", {})
            )
            
            return spec
        
        except Exception as e:
            print(f"[generate_multi_image_spec] Attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            time.sleep(RETRY_SLEEP)
    
    return None

def process_one_case_single_image(args_tuple):
    qid, entity, pool_sampler, task_config, max_iter, output_dir, use_edit, result_lists, global_lock, providers = args_tuple

    client_text = providers["text"].client
    provider_image = providers["image"]
    provider_check = providers["check"]

    current_entity = entity

    while True:
        if current_entity is None:
            with global_lock:
                current_entity = pool_sampler.sample(task_config)

            if current_entity is None:
                print(f"[qid {qid}] FAILED: Pool completely exhausted, cannot continue")
                return None

        triplet = None
        for attempt in range(1, 4):
            try:
                triplet = generate_one_case(task_config, current_entity, client_text)
                if triplet:
                    break
            except Exception as e:
                print(f"[qid {qid}] Generate case attempt {attempt} failed: {e}")

        if not triplet:
            current_entity = None
            continue

        statement = generate_statement_from_qa(triplet["question"], triplet["answer"], client_text)
        image_path = os.path.join(output_dir, f"q{qid:05d}.png")

        success = False
        for outer in range(1, max_iter + 1):
            try:
                last_url = generate_image(f"generate an image: {triplet['prompt']}", provider_image)

                if check_image_consistency(statement, last_url, provider_check):
                    save_image(download_bytes(last_url), image_path)
                    success = True
                    break
                else:
                    print(f"[qid {qid}] Verification FAILED on attempt {outer} (generated image)")
                    print(f"  Statement: {statement}")
                    print(f"  Image URL: {last_url[:100]}..." if len(last_url) > 100 else f"  Image URL: {last_url}")
                    if use_edit:
                        try:
                            bytes_in = download_bytes(last_url)
                            edit_p = f"Make the image exactly match: '{statement}'. Only modify what's needed."
                            last_url = edit_image_get_url(bytes_in, edit_p, provider_image)
                            if check_image_consistency(statement, last_url, provider_check):
                                save_image(download_bytes(last_url), image_path)
                                success = True
                                break
                            else:
                                print(f"[qid {qid}] Verification FAILED on attempt {outer} (edited image)")
                                print(f"  Statement: {statement}")
                                print(f"  Image URL: {last_url[:100]}..." if len(last_url) > 100 else f"  Image URL: {last_url}")
                        except Exception as e:
                            print(f"[qid {qid}] Edit ERROR on attempt {outer}: {e}")

            except Exception as e:
                print(f"[qid {qid}] Generation ERROR on attempt {outer}: {e}")

        if success:
            result_lists["triplets"][qid] = triplet
            result_lists["paths"][qid] = [image_path]
            result_lists["prompts"].append({
                "qid": qid,
                "prompt": triplet["prompt"],
                "question": triplet["question"],
                "answer": triplet["answer"],
                "metadata": triplet.get("metadata", {})
            })
            result_lists["statements"].append({
                "qid": qid,
                "statement": statement,
                "question": triplet["question"],
                "answer": triplet["answer"]
            })

            return {"qid": qid, "success": True}
        else:
            print(f"[qid {qid}] Image generation/verification failed after {max_iter} attempts, trying new entity...")
            current_entity = None

def execute_multi_image_spec(spec: MultiImageSpec, provider_image: ProviderRuntime, max_retries: int = 3) -> Optional[List[str]]:
    image_urls = []
    image_store = {}

    for ip in spec.image_prompts:
        try:
            if ip.type == "generate":
                url = generate_image(f"generate an image: {ip.prompt}", provider_image, max_retries=max_retries)
                image_urls.append(url)
                image_store[ip.index] = url

            elif ip.type == "edit":
                if ip.ref_image_index is None or ip.ref_image_index not in image_store:
                    print(f"[execute] ERROR: edit image {ip.index} references invalid ref_image_index {ip.ref_image_index}")
                    return None

                ref_url = image_store[ip.ref_image_index]
                ref_bytes = download_bytes(ref_url)
                edit_url = edit_image_get_url(ref_bytes, f"edit: {ip.prompt}", provider_image, max_retries=max_retries)
                image_urls.append(edit_url)
                image_store[ip.index] = edit_url

            else:
                print(f"[execute] ERROR: Unknown image_prompt type: {ip.type}")
                return None

        except Exception as e:
            print(f"[execute] ERROR at image {ip.index}: {e}")
            return None

    return image_urls

def process_one_case_multi_image(args_tuple):
    qid, entity, pool_sampler, task_config, max_iter, output_dir, use_edit, result_lists, global_lock, providers = args_tuple

    client_text = providers["text"].client
    provider_image = providers["image"]
    provider_check = providers["check"]

    current_entity = entity

    while True:
        if current_entity is None:
            with global_lock:
                current_entity = pool_sampler.sample(task_config)

            if current_entity is None:
                print(f"[qid {qid}] FAILED: Pool completely exhausted (multi-image)")
                return None

        spec = None
        for attempt in range(1, 4):
            try:
                spec = generate_multi_image_spec(task_config, current_entity, client_text)
                if spec:
                    break
            except Exception as e:
                print(f"[qid {qid}] Generate multi-image spec attempt {attempt} failed: {e}")

        if not spec:
            print(f"[qid {qid}] Failed to generate spec, trying new entity...")
            current_entity = None
            continue

        statement = generate_statement_from_qa(spec.question, spec.answer, client_text)

        success = False
        for outer in range(1, max_iter + 1):
            try:
                image_urls = execute_multi_image_spec(spec, provider_image)
                if image_urls is None:
                    print(f"[qid {qid}] Image generation/edit ERROR on attempt {outer}")
                    continue

                if verify_multi_image_consistency(statement, image_urls, provider_check):
                    image_paths = []
                    for img_idx, img_url in enumerate(image_urls):
                        img_path = os.path.join(output_dir, f"q{qid:05d}_{img_idx}.png")
                        save_image(download_bytes(img_url), img_path)
                        image_paths.append(img_path)

                    success = True
                    break
                else:
                    print(f"[qid {qid}] Verification FAILED on attempt {outer}")
                    print(f"  Statement: {statement}")
                    for idx, img_url in enumerate(image_urls):
                        url_preview = img_url[:100] + "..." if len(img_url) > 100 else img_url
                        print(f"  Image {idx} URL: {url_preview}")

            except Exception as e:
                print(f"[qid {qid}] ERROR on attempt {outer}: {e}")

        if success:
            result_lists["multi_specs"][qid] = spec.to_dict()
            result_lists["paths"][qid] = image_paths

            result_lists["prompts"].append({
                "qid": qid,
                "image_prompts": [
                    {
                        "index": ip.index,
                        "type": ip.type,
                        "prompt": ip.prompt,
                        "ref_image_index": ip.ref_image_index
                    }
                    for ip in spec.image_prompts
                ],
                "question": spec.question,
                "answer": spec.answer,
                "metadata": spec.metadata
            })

            result_lists["statements"].append({
                "qid": qid,
                "statement": statement,
                "question": spec.question,
                "answer": spec.answer,
                "num_images": len(image_urls)
            })

            return {"qid": qid, "success": True}
        else:
            print(f"[qid {qid}] Image verification failed after {max_iter} attempts, trying new entity...")
            current_entity = None

def build_annotations(ids: List[int], paths: Dict[int, List[str]], triplets: Dict[int, Dict], output_path: str, mode: str = "single"):
    
    result = []
    
    if mode == "single":
        for qid in ids:
            triplet = triplets.get(qid, {})
            img_path = paths.get(qid, [None])[0]
            
            messages = [
                {"role": "user", "content": f"<image>\n{triplet.get('question', '')}"},
                {"role": "assistant", "content": triplet.get("answer", "")}
            ]
            
            item = {
                "messages": messages,
                "images": [img_path] if img_path else [],
                "qid": qid,
                "prompt": triplet.get("prompt", ""),
                "metadata": triplet.get("metadata", {})
            }
            result.append(item)
    
    elif mode == "multi":
        for qid in ids:
            multi_spec = triplets.get(qid, {})
            img_paths = paths.get(qid, [])
            
            messages = [
                {"role": "user", "content": f"<images>\n{multi_spec.get('question', '')}"},
                {"role": "assistant", "content": multi_spec.get("answer", "")}
            ]
            
            item = {
                "messages": messages,
                "images": img_paths,
                "qid": qid,
                "image_prompts": [
                    {
                        "index": ip.get("index"),
                        "type": ip.get("type"),
                        "prompt": ip.get("prompt"),
                        "ref_image_index": ip.get("ref_image_index")
                    }
                    for ip in multi_spec.get("image_prompts", [])
                ],
                "metadata": multi_spec.get("metadata", {})
            }
            result.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Annotation saved: {output_path} ({len(result)} cases, mode={mode})")

def main():
    parser = argparse.ArgumentParser(description="God Engine v4 - Complete VQA Data Generation")
    
    parser.add_argument('--task', help='Task short description (if no config file)')
    parser.add_argument('--config', help='Path to JSON config file')
    parser.add_argument('--save_config_template', help='Save config template to this path')
    parser.add_argument('--api_config', help='Path to API config JSON (see config.example.json)')
    
    parser.add_argument('--num', type=int, default=100, help='Number of cases to generate')
    parser.add_argument('--mode', choices=['single', 'multi'], default='single', help='Image mode')
    parser.add_argument('--num_objects', type=int, default=1, help='Number of objects per case')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images in multi mode')
    parser.add_argument('--multi_image_form', choices=['multi_generate', 'story_chain', 'mixed'], default='story_chain', help='Multi-image form')
    
    parser.add_argument('--objects_size', type=int, default=20, help='Size of auto-generated objects list')
    parser.add_argument('--attributes_size', type=int, default=20, help='Size of auto-generated attributes list')
    parser.add_argument('--scenes_size', type=int, default=20, help='Size of auto-generated scenes list')
    parser.add_argument('--styles_size', type=int, default=20, help='Size of auto-generated styles list')
    parser.add_argument('--max_items_per_call', type=int, default=MAX_ITEMS_PER_LLM_CALL, help='Max items per LLM call for pool generation (avoid quality degradation)')
    parser.add_argument('--llm_decide_attr_size', action='store_true', default=False, help='Use LLM to decide optimal attribute pool size instead of using --attributes_size')
    parser.add_argument('--objects', nargs='+', help='Custom object list')
    parser.add_argument('--attributes', nargs='+', help='Custom attributes list')
    parser.add_argument('--scenes', nargs='+', help='Custom scenes list')
    parser.add_argument('--styles', nargs='+', help='Custom styles list')
    parser.add_argument('--global_pool', help='Path to global pool JSON file')
    parser.add_argument('--generate_missing', action='store_true', default=True, help='Auto-generate missing lists')
    
    parser.add_argument('--max_iter', type=int, default=3, help='Max generation attempts per case')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    parser.add_argument('--annotation_output', default='annotations.json')
    parser.add_argument('--prompts_output', default='prompts.jsonl')
    parser.add_argument('--statements_output', default='statements.jsonl')
    parser.add_argument('--pool_output', default='pool.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--use_edit', action='store_true', default=False, 
                    help='Enable image editing repair after failed verification (within max_iter attempts)')
    
    args = parser.parse_args()
    
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return

    api_cfg = load_api_config(args.api_config)
    providers = build_providers(api_cfg)
    if "text" not in providers or "image" not in providers or "check" not in providers:
        raise ValueError("API config must define providers: text, image, check")
    if "embed" not in providers:
        providers["embed"] = providers["text"]

    global MODEL_TEXT, MODEL_IMAGE, MODEL_CHECK
    MODEL_TEXT = providers["text"].model
    MODEL_IMAGE = providers["image"].model
    MODEL_CHECK = providers["check"].model
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for p in [args.prompts_output, args.statements_output]:
        if os.path.exists(p):
            os.remove(p)
    
    if args.config:
        print(f"Loading config from: {args.config}")
        task_config = load_config_from_json(args.config)
    else:
        print("Building config from command-line arguments")
        pool_cfg = PoolConfig(
            objects=args.objects,
            attributes=args.attributes,
            scenes=args.scenes,
            styles=args.styles,
            generate_missing=args.generate_missing,
            use_global_pool=args.global_pool
        )
        task_config = TaskConfig(
            task_id=args.task or "default",
            description=args.task or "VQA task",
            mode=args.mode,
            num_objects=args.num_objects,
            num_images=args.num_images,
            multi_image_form=args.multi_image_form,
            pool_config=pool_cfg
        )
    
    print(f"\n{'='*60}")
    print(f"Task: {task_config.task_id}")
    print(f"Description: {task_config.description}")
    print(f"Mode: {task_config.mode} | Objects: {task_config.num_objects} | Images: {task_config.num_images}")
    if task_config.mode == "multi":
        print(f"Multi-image form: {task_config.multi_image_form}")
    print(f"{'='*60}\n")
    
    client_main = providers["text"].client
    pool_cfg = build_adaptive_pool(
        task_config, client_main,
        objects_size=args.objects_size,
        attributes_size=args.attributes_size,
        scenes_size=args.scenes_size,
        styles_size=args.styles_size,
        max_items_per_call=args.max_items_per_call,
        llm_decide_attr_size=args.llm_decide_attr_size
    )
    task_config.pool_config = pool_cfg
    
    pool_data = {
        "objects": pool_cfg.objects or [],
        "attributes": pool_cfg.attributes or [],
        "scenes": pool_cfg.scenes or [],
        "styles": pool_cfg.styles or [],
        "custom_attributes": pool_cfg.custom_attributes
    }
    with open(os.path.join(args.output_dir, args.pool_output), 'w', encoding='utf-8') as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)
    print(f"Pool saved to: {os.path.join(args.output_dir, args.pool_output)}\n")
    
    combination_pool = build_combination_pool(pool_cfg, task_config)
    if not combination_pool:
        print("ERROR: No combinations generated!")
        return
    
    global_lock = threading.Lock()
    result_dicts = {
        "triplets": {},
        "multi_specs": {},
        "paths": {},
        "prompts": [],
        "statements": []
    }
    
    print(f"Pre-allocating {args.num} entities...")
    print(f"  Pool size: {len(combination_pool):,}")
    
    #pool_sampler = PoolSampler(combination_pool)
    pool_sampler = PoolSampler(combination_pool, pool_cfg)

    tasks = []
    for qid in range(1, args.num + 1):
        entity = pool_sampler.sample(task_config)
        if entity is None:
            print(f"[pool] Exhausted at qid {qid}, pre-allocated {len(tasks)}/{args.num} tasks")
            break
        #tasks.append((qid, entity, pool_sampler, task_config, args.max_iter, args.output_dir, result_dicts, global_lock))
        tasks.append((qid, entity, pool_sampler, task_config, args.max_iter, 
              args.output_dir, args.use_edit, result_dicts, global_lock, providers))
    
    print(f"[pool] Pre-allocated {len(tasks)}/{args.num} tasks\n")
    
    process_func = process_one_case_multi_image if task_config.mode == "multi" else process_one_case_single_image
    
    print(f"Generating {len(tasks)} cases with {args.parallel} parallel workers... (mode={task_config.mode})\n")
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_func, task) for task in tasks]
        
        with tqdm(total=len(tasks), desc="Successful cases") as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        pbar.update(1)
                except Exception as e:
                    print(f"Worker error: {e}")
    
    save_jsonl(result_dicts["prompts"], os.path.join(args.output_dir, args.prompts_output))
    save_jsonl(result_dicts["statements"], os.path.join(args.output_dir, args.statements_output))
    
    triplets_to_use = result_dicts["multi_specs"] if task_config.mode == "multi" else result_dicts["triplets"]
    
    build_annotations(
        list(triplets_to_use.keys()),
        result_dicts["paths"],
        triplets_to_use,
        os.path.join(args.output_dir, args.annotation_output),
        mode=task_config.mode
    )
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Total cases: {len(triplets_to_use)}")
    print(f"Mode: {task_config.mode}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
