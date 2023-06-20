import struct
import sys


def f16_bytes_to_float(b):
    """Convert half-precision floating point bytes to float

    - Supports signed zero and denormals-as-zero (DAZ)
    - Supports infinity and NaN

    Args:
        b (bytes): Half-precision bytes
    """
    # Pad to 32-bit boundary
    if sys.byteorder == 'little':
        b += b'\x00' * 2  
    else:
        b = b'\x00' * 2 + b
    i = struct.unpack('<I', b)[0]

    t1 = i & 0x7fff  # Non-sign bits
    t2 = i & 0x8000  # Sign bit
    t3 = i & 0x7c00  # Exponent bits

    t1 <<= 13  # Align mantissa on MSB
    t2 <<= 16  # Shift sign bit into position

    t1 += 0x38000000  # Adjust bias

    t1 = t1 if t3 else 0  # Denormals-as-zero

    t1 |= t2  # Re-insert sign bit

    return struct.unpack('f', struct.pack('I', t1))[0]


def float_to_f16_bytes(f):
    """Convert float to half-precision floating point bytes

    - Supports signed zero, denormals-as-zero (DAZ), flush-to-zero (FTZ), clamp-to-max
    - Does not support infinities or NaN

    Args:
        f (bytes): 
    """
    i = struct.unpack('I', struct.pack('f', f))[0]

    t1 = i & 0x7fffffff  # Non-sign bits
    t2 = i & 0x80000000  # Sign bit
    t3 = i & 0x7f800000  # Exponent bits

    t1 >>= 13  # Align mantissa on MSB
    t2 >>= 16  # Shift sign bit into position

    t1 -= 0x1c000  # Adjust bias

    t1 = t1 if t3 >= 0x38800000 else 0  # Flush-to-zero
    t1 = t1 if t3 <= 0x8e000000 else 0x7bff  # Clamp-to-max
    t1 = t1 if t3 else 0  # Denormals-as-zero

    t1 |= t2  # Re-insert sign bit

    return struct.pack('<I', t1)[:2]


def test(f):
    b = float_to_f16_bytes(f)
    print(b)
    print(f16_bytes_to_float(b))
    

if __name__ == '__main__':
    test(0.001)
    test(0.05)
    test(0.0)
    test(10.0)
    test(100.0)
    test(1000.0)

    test(-1.5)
    test(-100.0)
    test(-1000.0)