import torch
from .buffermanager import BufferManager

class BitOps:

    def __init__(self, L: int, max_elements: int = None, device = "cpu"):
        self.device = device
        self.L = L
        self.dim = 2 ** L if max_elements is None else max_elements
        self.manager = BufferManager.get_manager(self.dim, device, self.set_dtype())


    def set_dtype(self):
        """
        Determines the appropriate integer data type (dtype) to represent values with L bits.

        Parameters:
            None. Uses the instance attribute `self.L`.

        Returns:
            torch.dtype: Returns `torch.int32` if L <= 31, otherwise returns `torch.int64`.
        """

        if self.L < 32:
            return torch.int32
        else:
            return torch.int64


    def set_bits(self, num, bits, out=None):
        """
        Sets one or more bits to 1 in an integer or each element of a tensor.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers where the bits will be set.
            bits (int or list of int): Index or list of indices of the bits to set (0 = least significant bit).
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer with the specified bits set to 1.
                - If `num` is a tensor, returns `out` containing the modified values.
        """

        if isinstance(bits, int):
            mask = 1 << bits
        else:
            mask = 0
            for bit in bits:
                mask |= 1 << bit


        if isinstance(num, int):
            return num | mask

        if out is None:
            out = torch.empty_like(num)


        num_elements = len(num)
        nout = out[:num_elements]

        torch.bitwise_or(num, mask, out=nout)
        return out


    def clear_bits(self, num, bits, out=None):
        """
        Clears (sets to 0) one or more bits in an integer or each element of a tensor.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers where bits will be cleared.
            bits (int or list of int): Index or list of indices of the bits to clear (0 = least significant bit).
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer with the specified bits cleared.
                - If `num` is a tensor, returns `out` containing the modified values.
        """

        if isinstance(bits, int):
            mask = ~(1 << bits)
        else:
            mask = 0
            for bit in bits:
                mask |= 1 << bit
            mask = ~mask

        if isinstance(num, int):
            return num & mask

        if out is None:
            out = torch.empty_like(num)

        num_elements = len(num)
        nout = out[:num_elements]

        torch.bitwise_and(num, mask, out=nout)
        return out


    def get_bit(self, num, bit_i: int, out=None):
        """
        Retrieves the value of the bit at position `bit_i` from an integer or each element of a tensor.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers.
            bit_i (int): Index of the bit to retrieve (0 = least significant bit).
            out (torch.Tensor, optional): Output tensor to store the resulting bits.
                If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the value of the bit (0 or 1).
                - If `num` is a tensor, returns `out` containing the extracted bit values.
        """

        if isinstance(num, int):
            return (num >> bit_i) & 1

        if out is None:
            out = torch.empty_like(num)

        num_elements = len(num)
        nout = out[:num_elements]

        torch.bitwise_right_shift(num, bit_i, out=nout)
        nout.bitwise_and_(1)
        return out


    def count_bits(self, num, out=None):
        """
        Counts the number of set bits (1s) in an integer or in each element of a tensor.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers whose set bits are to be counted.
            out (torch.Tensor, optional): Output tensor to store the bit counts.
                If None, a new tensor filled with zeros is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the number of bits set to 1.
                - If `num` is a tensor, returns `out` containing the bit counts for each element.
        """

        if isinstance(num, int):
            uns = 0
            for i in range(self.L):
                uns += (num >> i) & 1
            return uns


        if out is None:
            out = torch.zeros_like(num)
        else:
            out.zero_()

        tmp = self.manager.get()

        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        for i in range(self.L):
            torch.bitwise_right_shift(num, i, out=ntmp)
            ntmp.bitwise_and_(1)
            nout.add_(ntmp)

        self.manager.release(tmp)
        return out


    def permute_bits(self, num, bit_i: int, bit_j: int, out=None):
        """
        Swaps (permutes) the bits at positions `bit_i` and `bit_j` in an integer or each element of a tensor.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers where bits will be swapped.
            bit_i (int): Index of the first bit to swap (0 = least significant bit).
            bit_j (int): Index of the second bit to swap.
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer with bits at `bit_i` and `bit_j` swapped.
                - If `num` is a tensor, returns `out` containing the modified values.
        """

        if isinstance(num, int):
            mask = ((num >> bit_i) ^ (num >> bit_j)) & 1
            return num ^ ((mask << bit_i) | (mask << bit_j))


        if out is None:
            out = torch.empty_like(num)

        if bit_j < bit_i:
            bit_i, bit_j = bit_j, bit_i

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        torch.bitwise_right_shift(num, bit_j - bit_i, out=ntmp)
        ntmp.bitwise_xor_(num)
        ntmp.bitwise_right_shift_(bit_i)
        ntmp.bitwise_and_(1) # ntmp = mask

        torch.bitwise_right_shift(ntmp, bit_j, out=nout) # nout = mask << j
        ntmp.bitwise_right_shift_(bit_i) # ntmp = mask << i
        nout.bitwise_or_(ntmp) # nout = (mask << i) | (mask << j)
        nout.bitwise_xor_(num)

        self.manager.release(tmp)
        return out


    def rotate_bits(self, num, positions: int = 1, out=None):
        """
        Rotates the bits of an integer or each element of a tensor to the left by `positions` positions.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers to rotate.
            positions (int, optional): Number of positions to rotate to the left. Defaults to 1.
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer after left bit rotation.
                - If `num` is a tensor, returns `out` containing the rotated values.
        """

        positions %= self.L
        mask = (1 << self.L) - 1

        if isinstance(num, int):
            return (num >> (self.L - positions)) | ((num << positions) & mask)


        if out is None:
            out = torch.empty_like(num)

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        torch.bitwise_left_shift(num, positions, out=ntmp)


        torch.bitwise_right_shift(num, self.L - positions, out=nout)
        ntmp.bitwise_and_(mask)
        nout.bitwise_or_(ntmp)

        self.manager.release(tmp)
        return out


    def flip_bits(self, num, bits = None, out=None):
        """
        Flips (inverts) one or more bits in an integer or each element of a tensor.
        If `bits` is None, all bits are flipped.

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers whose bits will be flipped.
            bits (int or list of int, optional): Index or list of indices of bits to flip (0 = least significant bit).
                If None, all bits are flipped.
            out (torch.Tensor, optional): Output tensor to store the result.
                If None, a new tensor is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer with specified bits flipped.
                - If `num` is a tensor, returns `out` containing the modified values.
        """

        if bits is None:
            mask = (1 << self.L) - 1
        elif isinstance(bits, int):
            mask = 1 << bits
        else:
            mask = 0
            for bit in bits:
                mask |= 1 << bit


        if isinstance(num, int):
            return num ^ mask

        if out is None:
            out = torch.empty_like(num)

        num_elements = len(num)
        nout = out[:num_elements]

        torch.bitwise_xor(num, mask, out=nout)
        return out


    def reverse_bits(self, num, out=None):
        """
        Reverses the order of bits in an integer or each element of a tensor
        (mirrors the bitstring around its center).

        Parameters:
            num (int or torch.Tensor): Input integer or tensor of integers to reverse bits.
            out (torch.Tensor, optional): Output tensor to store the result.
                If None, a new tensor filled with zeros is created.

        Returns:
            int or torch.Tensor:
                - If `num` is an int, returns the integer with bits reversed.
                - If `num` is a tensor, returns `out` containing the bit-reversed values.
        """

        if isinstance(num, int):
            n1 = 0
            for i in range(self.L):
                n1 |= ((num >> i) & 1) << (self.L - 1 - i)
            return n1


        if out is None:
            out = torch.zeros_like(num)
        else:
            out.zero_()

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        for i in range(self.L):
            torch.bitwise_right_shift(num, i, out=ntmp)
            ntmp.bitwise_and_(1)
            ntmp.bitwise_left_shift_(self.L-1-i)
            nout.bitwise_or_(ntmp)

        self.manager.release(tmp)
        return out


    def xor_bits(self, num, bits=None, out=None):
        """
        Computes the parity (XOR-sum) of specific bits in an integer tensor.

        Parameters:
            num (torch.Tensor): Input tensor of integers.
            bits (int or list of int): Bit positions to extract and XOR together.
            out (torch.Tensor, optional): Output tensor to store the result.
                                        Must have the same shape as `num`.
                                        If None, a new tensor is allocated.

        Returns:
            torch.Tensor: Tensor where each element is 0 or 1, representing the XOR
                        of the specified bits of the corresponding element in `num`.

        Notes:
            - The operation is done in-place on `out` if provided.
            - `bits` can be a single integer or a list of bit positions.
            - This effectively computes the parity of the selected bits.
        """

        if out is None:
            out = torch.zeros_like(num)
        else:
            out.zero_()

        if bits is None:
            bits = range(self.L)
        elif isinstance(bits, int):
            bits = [bits]

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        for bit in bits:
            torch.bitwise_right_shift(num, bit, out=ntmp)

            nout.bitwise_xor_(ntmp)
        nout.bitwise_and_(1)

        self.manager.release(tmp)
        return out


    def and_bits(self, num, bits=None, out=None):
        """
        Computes the AND of specific bits in an integer tensor.

        Parameters:
            num (torch.Tensor): Input tensor of integers.
            bits (int or list of int): Bit positions to extract and AND together.
            out (torch.Tensor, optional): Output tensor for the result.

        Returns:
            torch.Tensor: Tensor where each element is 0 or 1, the AND of the specified bits.
        """

        if out is None:
            out = torch.zeros_like(num)
        else:
            out.zero_()

        if bits is None:
            bits = range(self.L)
        elif isinstance(bits, int):
            bits = [bits]

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        for bit in bits:
            torch.bitwise_right_shift(num, bit, out=ntmp)

            nout.bitwise_and_(ntmp)
        nout.bitwise_and_(1)

        self.manager.release(tmp)
        return out


    def or_bits(self, num, bits=None, out=None):
        """
        Computes the OR of specific bits in an integer tensor.

        Parameters:
            num (torch.Tensor): Input tensor of integers.
            bits (int or list of int): Bit positions to extract and AND together.
            out (torch.Tensor, optional): Output tensor for the result.

        Returns:
            torch.Tensor: Tensor where each element is 0 or 1, the AND of the specified bits.
        """

        if out is None:
            out = torch.zeros_like(num)
        else:
            out.zero_()

        if bits is None:
            bits = range(self.L)
        elif isinstance(bits, int):
            bits = [bits]

        tmp = self.manager.get()
        num_elements = len(num)
        nout = out[:num_elements]
        ntmp = tmp[:num_elements]

        for bit in bits:
            torch.bitwise_right_shift(num, bit, out=ntmp)

            nout.bitwise_or_(ntmp)
        nout.bitwise_and_(1)

        self.manager.release(tmp)

        return out
