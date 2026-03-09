import torch

class BufferManager:
    """
    Manages a set of reusable tensors (buffers) with a global registry to allow
    sharing across multiple instances with the same dimension, device, and dtype.

    Each buffer has a flag indicating whether it is in use. If all buffers are
    occupied, a new one is created on demand. The global registry ensures that
    multiple classes or simulators requesting buffers with the same properties
    will reuse the same BufferManager instance.
    """
    
    _registry = {}
    _index_registry = {}

    def __init__(self, dim, device="cpu", dtype=torch.complex64):
        """
        Initializes the buffer manager for a given dimension, device, and dtype.

        Args:
            dim (int): dimension of each tensor.
            device (str or torch.device, optional): device where tensors will be allocated.
            dtype (torch.dtype, optional): data type of the tensors.
        """
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype
        self.buffers = []
        self.in_use = []


    @classmethod
    def get_manager(cls, dim, device, dtype):
    
        key = (dim, str(device), dtype)
        if key in cls._registry:
            return cls._registry[key]

        dim = int(dim)
        dim = 1 << (dim - 1).bit_length()
    
        same_device_dtype = [
            (d, dev, dt) for (d, dev, dt) in cls._registry.keys()
            if dev == str(device) and dt == dtype and d == dim
        ]
        if same_device_dtype:
            d_closest = min(same_device_dtype, key=lambda k: k[0])
            return cls._registry[d_closest]
    
        next_pow2 = dim
        new_key = (dim, str(device), dtype)
        cls._registry[new_key] = cls(dim, device, dtype)
        return cls._registry[new_key]

    @classmethod
    def delete_manager(cls, dim, device="cpu", dtype=torch.complex64):
        """
        Delete a BufferManager from the registry for the given key.

        Parameters
        ----------
        dim : int
            Dimension of the buffers handled by the manager.
        device : str or torch.device
            Device where the buffers are allocated.
        dtype : torch.dtype
            Data type of the buffers.
        """
        key = (dim, str(device), dtype)
        if key in cls._registry:
            cls._registry[key].clear()
            del cls._registry[key]
        else:
            raise KeyError(f"No BufferManager found for key {key}")


    @classmethod
    def get_index(cls, dim, device="cpu"):
        """
        Returns an immutable index tensor [0, 1, ..., dim-1] from the registry.
        If it does not exist, creates it.

        Args:
            dim (int): length of the index tensor.
            device (str or torch.device): device for tensor allocation.
            dtype (torch.dtype): data type of the tensor.

        Returns:
            torch.Tensor: immutable index tensor.
        """
        dtype = torch.int32 if dim < 2 ** 32 else torch.int64
        key = (dim, str(device))
        if key not in cls._index_registry:
            cls._index_registry[key] = torch.arange(dim, device=device, dtype=dtype)
        return cls._index_registry[key]

    
    def get(self):
        """
        Returns an available buffer. If none is free, creates a new one.

        Returns:
            torch.Tensor: a tensor ready for use.
        """
        for i, used in enumerate(self.in_use):
            if not used:
                self.in_use[i] = True
                return self.buffers[i]

        buf = torch.empty(self.dim, device=self.device, dtype=self.dtype)
        self.buffers.append(buf)
        self.in_use.append(True)
        return buf

    def release(self, buf):
        """
        Marks a buffer as available, allowing it to be reused.

        Args:
            buf (torch.Tensor): tensor previously obtained via `get()`.

        Raises:
            ValueError: if the buffer does not belong to this manager.
        """
        for i, b in enumerate(self.buffers):
            if b is buf:
                self.in_use[i] = False
                return
        raise ValueError("Buffer does not belong to this manager")


    def acquire_all(self):
        """
        Marks all existing buffers as in use.

        Returns:
            list[torch.Tensor]: list of all buffers now marked as in use.
        """

        for i in range(len(self.in_use)):
            self.in_use[i] = True

        return self.buffers

    def release_all(self):
        """
        Marks all buffers as available for reuse.

        Returns:
            list[torch.Tensor]: list of all buffers now marked as free.
        """
        for i in range(len(self.in_use)):
            self.in_use[i] = False

        return self.buffers

    
    def clear(self):
        """
        Deallocates all buffers managed by this instance.

        This removes all stored tensors and resets the usage flags.
        After calling this method, the BufferManager will behave as if
        no buffers had been allocated.

        Returns:
            None
        """
        self.buffers.clear()
        self.in_use.clear()
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
