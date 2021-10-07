from core import ImageSegmentation

if __name__ == '__main__':
    IS = ImageSegmentation('segment.jpg', show_results=True, save_results=True)
    IS.run()

