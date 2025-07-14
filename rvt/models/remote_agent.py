import argparse
import os

from omegaconf import OmegaConf
import torch
try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass
import logging
import time
from typing import Dict, Optional, Tuple
import websockets.sync.client
import functools
import msgpack
import numpy as np

import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import rvt.mvt.config as default_mvt_cfg
import rvt.models.rvt_agent as rvt_agent
import rvt.config as default_exp_cfg

from rvt.mvt.mvt import MVT

from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)

from rvt.utils.rvt_utils import load_agent as load_agent_state
import os 
from yarr.agents.agent import ActResult
from rvt.utils.t5_encoder import T5Embedder

def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
    lang_type='clip',
):
    device = f"cuda:{device}"
    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()


    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    if lang_type == 't5':
        mvt_cfg.add_lang_t5 = True
    else:
        mvt_cfg.add_lang_t5 = False
    if lang_type == 'clip':
        mvt_cfg.add_lang = True
    else:
        mvt_cfg.add_lang = False
    mvt_cfg.freeze()

    # for rvt-2 we do not change place_with_mean regardless of the arg
    # done this way to ensure backward compatibility and allow the
    # flexibility for rvt-1
    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    rvt = MVT(
        renderer_device=device,
        **mvt_cfg,
    )
    t5_embedder = None
    if lang_type == 't5':
        t5_embedder = T5Embedder(from_pretrained="google/t5-v1_1-xxl", model_max_length=77, device=device, local_files_only=True)

    agent = rvt_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        add_lang=mvt_cfg.add_lang,
        add_lang_t5=mvt_cfg.add_lang_t5,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        t5_embedder=t5_embedder,
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )


    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()

    # print("Agent Information")
    # print(agent)
    return agent


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class WebsocketClientPolicyAgent():
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def act(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)


def process_obs(obs: Dict, device: str = 'cuda:0') -> Dict:
    # convert to torch tensor except key 'lang_goal', 'lang_goal_bbox'
    prepped_data = {}
    for k, v in obs.items():
        if k not in ['lang_goal', 'lang_goal_bbox', 'visualize', 'visual_prompt_type', 'visualize_save_dir'] and obs[k] is not None:
            prepped_data[k] = torch.tensor(v, device=device)
    prepped_data['lang_goal'] = obs['lang_goal']
    prepped_data['lang_goal_bbox'] = obs['lang_goal_bbox']
    return prepped_data

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: rvt_agent.RVTAgent,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                prepped_data = process_obs(obs, device=self._policy._device)

                infer_time = time.monotonic()
                action_result = self._policy.act(-1, prepped_data, deterministic=True, \
                visualize=obs.get("visualize", False), \
                visual_prompt_type=obs.get("visual_prompt_type", []), \
                visualize_save_dir=obs.get("visualize_save_dir", ""))
                print(action_result)
                action = {}
                action['action'] = action_result.to_dict()

                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--exp-cfg-path", type=str, default=None)
    parser.add_argument("--mvt-cfg-path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--use-input-place-with-mean", action="store_true", default=False)
    parser.add_argument("--lang-type", type=str, default="clip")
    args = parser.parse_args()

    model_paths = []
    model_paths.append(os.path.join(args.model_folder, args.model_name))

    agent = load_agent(
        model_path=model_paths[0],
        exp_cfg_path=args.exp_cfg_path,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=".",
        device=args.device,
        use_input_place_with_mean=args.use_input_place_with_mean,
        lang_type=args.lang_type,
    )
    agent.eval()
    if isinstance(agent, rvt_agent.RVTAgent):
        agent.load_clip()

    WebsocketPolicyServer(agent, host="114.212.189.99", port=8000).serve_forever()