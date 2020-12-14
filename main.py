import sys
from termcolor import cprint

try:
    import A1.model
    import A2.model
    import B1.model
    import B2.model
except ImportError:
    print("Ml Models Not Found")
    sys.exit(1)



def run_mode(cls, dataset, root="."):
    model = cls(root, dataset)
    model.validate()
    model.predict()
    model.predict(extra=True)


def main():
    cprint("------------------- Model A1 ------------------", "cyan")
    run_mode(A1.model.Model, "celeba")
    cprint("------------------- Model A2 ------------------", "cyan")
    run_mode(A2.model.Model, "celeba")
    cprint("------------------- Model B1 ------------------", "cyan")
    run_mode(B1.model.Model, "cartoon_set")
    cprint("------------------- Model B2 ------------------", "cyan")
    run_mode(B2.model.Model, "cartoon_set")

if __name__ == "__main__":
    main()
