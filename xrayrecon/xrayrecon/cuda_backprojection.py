"""CUDA FFT filtering and a runtime-compiled parallel backprojection kernel.

Uses PyTorch's CUDA buffers/FFT/stream plus NVRTC and the CUDA driver API.
No system CUDA toolkit, C++ compiler, new Python dependency, or CPU fallback.
Optional CUDA libraries are loaded only after explicit CUDA selection.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import os
from pathlib import Path
import threading

import numpy as np

_KERNELS = {}
_COMPILE_LOCK = threading.Lock()


def _bind(library, name, arguments):
    function = getattr(library, name)
    function.restype = ct.c_int
    function.argtypes = arguments
    return function


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed (CUDA/NVRTC status {status})")


class _CompiledKernel:
    def __init__(self, torch, device):
        lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if os.name == "nt":
            candidates = sorted(lib_dir.glob("nvrtc64_*.dll"))
            candidates = [path for path in candidates if ".alt." not in path.name]
            if not candidates:
                raise RuntimeError("PyTorch's NVRTC DLL is missing")
            with os.add_dll_directory(str(lib_dir)):
                self.nvrtc = ct.CDLL(str(candidates[-1]))
            self.driver = ct.WinDLL("nvcuda.dll")
        else:
            candidates = list(lib_dir.glob("libnvrtc.so*"))
            candidates += list(lib_dir.parent.parent.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
            # CUDA 13 wheels put every library under nvidia/cu13/lib.
            candidates += list(lib_dir.parent.parent.glob("nvidia/cu*/lib/libnvrtc.so*"))
            library = str(sorted(candidates)[0]) if candidates else ctypes.util.find_library("nvrtc")
            if not library:
                raise RuntimeError("NVRTC shared library is missing from the CUDA installation")
            self.nvrtc = ct.CDLL(library)
            self.driver = ct.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")

        create = _bind(self.nvrtc, "nvrtcCreateProgram", [ct.POINTER(ct.c_void_p), ct.c_char_p,
                       ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p])
        compile_program = _bind(self.nvrtc, "nvrtcCompileProgram", [ct.c_void_p, ct.c_int,
                                ct.POINTER(ct.c_char_p)])
        destroy = _bind(self.nvrtc, "nvrtcDestroyProgram", [ct.POINTER(ct.c_void_p)])
        program = ct.c_void_p()
        source = Path(__file__).with_suffix(".cu").read_bytes()
        _check(create(ct.byref(program), source, b"cuda_backprojection.cu", 0, None, None),
               "nvrtcCreateProgram")
        try:
            major, minor = torch.cuda.get_device_capability(device)
            options = (ct.c_char_p * 3)(b"--std=c++11", b"--fmad=false",
                                       f"--gpu-architecture=compute_{major}{minor}".encode())
            status = compile_program(program, len(options), options)
            if status:
                size = ct.c_size_t()
                _bind(self.nvrtc, "nvrtcGetProgramLogSize", [ct.c_void_p, ct.POINTER(ct.c_size_t)])(program, ct.byref(size))
                log = ct.create_string_buffer(size.value)
                _bind(self.nvrtc, "nvrtcGetProgramLog", [ct.c_void_p, ct.c_void_p])(program, log)
                raise RuntimeError("CUDA backprojection compilation failed: " + log.value.decode())
            size = ct.c_size_t()
            _check(_bind(self.nvrtc, "nvrtcGetPTXSize", [ct.c_void_p, ct.POINTER(ct.c_size_t)])(program, ct.byref(size)), "nvrtcGetPTXSize")
            ptx = ct.create_string_buffer(size.value)
            _check(_bind(self.nvrtc, "nvrtcGetPTX", [ct.c_void_p, ct.c_void_p])(program, ptx), "nvrtcGetPTX")
        finally:
            destroy(ct.byref(program))
        self.module = ct.c_void_p()
        _check(_bind(self.driver, "cuModuleLoadData", [ct.POINTER(ct.c_void_p), ct.c_void_p])(
            ct.byref(self.module), ptx), "cuModuleLoadData")
        self.functions = {}
        get_function = _bind(self.driver, "cuModuleGetFunction", [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p])
        for name in ("fbp_float", "fbp_double"):
            function = ct.c_void_p()
            _check(get_function(ct.byref(function), self.module, name.encode()), "cuModuleGetFunction")
            self.functions[name] = function
        self.launch = _bind(self.driver, "cuLaunchKernel", [ct.c_void_p] + [ct.c_uint] * 7
                            + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p])

    def __call__(self, filtered, trig, output, stream):
        channels, angles, detectors = filtered.shape
        side = output.shape[-1]
        values = [ct.c_void_p(tensor.data_ptr()) for tensor in (filtered, trig, output)]
        values += [ct.c_int(value) for value in (detectors, angles, side, channels)]
        parameters = (ct.c_void_p * len(values))(*(ct.cast(ct.pointer(value), ct.c_void_p) for value in values))
        name = "fbp_float" if output.element_size() == 4 else "fbp_double"
        blocks = (output.numel() + 255) // 256
        _check(self.launch(self.functions[name], blocks, 1, 1, 256, 1, 1, 0,
                           ct.c_void_p(stream.cuda_stream), parameters, None), "cuLaunchKernel")


class CudaBackprojector:
    """Reusable FBP geometry; accepts (C, detector, angle) float32/64 sinograms.

    Supports the application's ramp-filtered and unfiltered, linear,
    circle=False backprojections. FFT filtering runs on CUDA. Temporary memory
    scales with sinogram and output sizes, not output pixels times angles.
    """
    def __init__(self, detector_size, theta_deg, *, output_size=None):
        theta = np.asarray(theta_deg, dtype=np.float64)
        if theta.ndim != 1 or not theta.size or not np.isfinite(theta).all():
            raise ValueError("theta_deg must be a nonempty finite 1-D array")
        if isinstance(detector_size, bool) or not isinstance(detector_size, (int, np.integer)) or detector_size < 1:
            raise ValueError("detector_size must be a positive integer")
        side = int(np.floor(np.sqrt(detector_size ** 2 / 2.0))) if output_size is None else output_size
        if isinstance(side, bool) or not isinstance(side, (int, np.integer)) or side < 1:
            raise ValueError("output_size must be a positive integer")
        self.detector_size, self.output_size = int(detector_size), int(side)
        self.theta_deg = theta.copy()
        self.padded_size = max(64, int(2 ** np.ceil(np.log2(2 * detector_size))))
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch reports no available CUDA device")
            self._torch = torch
            self.device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(self.device), _COMPILE_LOCK:
                # Establish PyTorch's primary context on this host thread before
                # calling the driver directly (device selection alone is lazy).
                torch.empty(0, device=self.device)
                torch.cuda.synchronize(self.device)
                key = self.device.index
                if key not in _KERNELS:
                    _KERNELS[key] = _CompiledKernel(torch, self.device)
                self._kernel = _KERNELS[key]
                radians = np.deg2rad(theta)
                self._trig = torch.as_tensor(np.column_stack((np.cos(radians), np.sin(radians))),
                                             dtype=torch.float64, device=self.device)
                # Match skimage's discrete ramp, including its DC bias correction.
                n = np.concatenate((np.arange(1, self.padded_size / 2 + 1, 2, dtype=int),
                                    np.arange(self.padded_size / 2 - 1, 0, -2, dtype=int)))
                impulse = np.zeros(self.padded_size)
                impulse[0], impulse[1::2] = 0.25, -1 / (np.pi * n) ** 2
                self._filter = 2 * torch.fft.fft(torch.as_tensor(impulse, device=self.device)).real
                torch.cuda.synchronize(self.device)
        except (ImportError, OSError, RuntimeError, AssertionError) as exc:
            raise RuntimeError(f"CUDA backprojection backend unavailable: {exc}") from exc

    def reconstruct(self, sinograms, *, filter_name="ramp", timings=None):
        """Return synchronized independent (C, S, S) NumPy reconstructions.

        Optional compute_ms covers GPU FFT/filtering/kernel execution and
        dispatch gaps, excluding explicit transfers and cached initialization.
        """
        sinograms = np.asarray(sinograms)
        if sinograms.ndim != 3 or sinograms.shape[0] < 1 or sinograms.shape[1:] != (self.detector_size, self.theta_deg.size):
            raise ValueError("sinograms must have shape (C, detector_size, angle_count), C > 0")
        if sinograms.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("sinograms must have dtype float32 or float64")
        if not np.isfinite(sinograms).all():
            raise ValueError("sinograms must contain only finite values")
        if filter_name not in ("ramp", None):
            raise ValueError("CUDA backprojection supports only ramp or None filters")
        torch = self._torch
        with torch.cuda.device(self.device), torch.no_grad():
            source = torch.from_numpy(np.array(sinograms, copy=True, order="C")).to(self.device)
            if timings is not None:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
            padded = torch.nn.functional.pad(source, (0, 0, 0, self.padded_size - self.detector_size))
            spectrum = torch.fft.fft(padded, dim=1)
            # skimage promotes the FFT result to complex128 via a float64 filter.
            spectrum = spectrum * (self._filter[None, :, None] if filter_name else
                                   torch.ones((1, self.padded_size, 1), dtype=torch.float64, device=self.device))
            filtered = torch.fft.ifft(spectrum, dim=1).real[:, :self.detector_size].permute(0, 2, 1).contiguous()
            output = torch.empty((source.shape[0], self.output_size, self.output_size),
                                 dtype=source.dtype, device=self.device)
            self._kernel(filtered, self._trig, output, torch.cuda.current_stream(self.device))
            if timings is not None:
                end.record()
            result = output.cpu().numpy()
            torch.cuda.synchronize(self.device)
            if timings is not None:
                timings["compute_ms"] = start.elapsed_time(end)
        return result
