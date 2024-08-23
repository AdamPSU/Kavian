NUM = {
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float16',
    'float32',
    'float64',
    'float128'
}

FLOAT = {
    'float16',
    'float32',
    'float64',
    'float128'
}

INT = {
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64'
}

CAT = {
    'object',
    'category',
    'string',
    'bool'
}

BOOL = {
    'bool'
}

DTYPE_PRIORITY = {
    'bool': 11,
    'object': 10,
    'category': 9,
    'string': 8,
    'float128': 7,
    'float64': 6,
    'float32': 5,
    'float16': 4,
    'int8': 3,
    'uint8': 3,
    'int16': 2,
    'uint16': 2,
    'int32': 1,
    'uint32': 1,
    'int64': 0,
    'uint64': 0
}