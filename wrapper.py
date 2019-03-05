from L1 import L1
from L2 import L2


def main():

    sp_object = L1()
    sp_object.get_image_vector()
    sp_object.find_reference()
    sequence_dir1_L1, sequence_dir2_L1 = sp_object.l1_distance_all()

    sp_object = L2()
    sp_object.get_image_vector()
    sp_object.find_reference()
    sequence_dir1_L2, sequence_dir2_L2 = sp_object.l2_distance_all()

    print("Aligning...")


if __name__ == "__main__":
    main()
