from jpeg_decoder.byte_N_bit_utils import pop_bit, read_byte
from jpeg_decoder.headers_reader import flags_names, flags_fun
from jpeg_decoder.plter import plt_mcu
from jpeg_decoder.scan_decoder import read_scan


def read_jpeg(file_name, scan=True, print_details=False):
    print('Reading file:', file_name)
    with open(file_name, 'rb') as img_file:
        headers = {}
        img = img_file.read()
        assert isinstance(img, bytes)
        ##
        popper = pop_bit(img)
        while True:
            b = read_byte(popper)
            if b == 0xff:
                flag_code = read_byte(popper)
                if flag_code in flags_names:
                    flag = 0xff00 | flag_code
                    print('Reading\t', hex(flag).upper(), '\t', flags_names[flag_code])
                    if callable(flags_fun[flag_code]):
                        if flag_code not in headers:
                            headers[flag_code] = []
                        headers[flag_code].append(
                            flags_fun[flag_code](flag_code, flags_names[flag_code], popper, print_details))
                    else:
                        print(flags_names[flag_code])

                    if flag_code == 0xda:
                        __img_array = read_scan(popper, headers, scan)
                        # break
                        return __img_array

                    continue


# todo short cut 4 file
filename = 'lena.jpg'
img_array = read_jpeg(filename, print_details=True)
plt_mcu(img_array)

print('-' * 20)
