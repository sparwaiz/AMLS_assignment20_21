import A1.model
import A2.model
import B1.model
import B2.model


def run_mode(cls, dataset, root="."):
    model = cls(root, dataset)
    model.validate()
    model.predict()
    model.predict(extra=True)


def main():
    run_mode(A1.model.Model, "celeba")
    run_mode(A2.model.Model, "celeba")
    run_mode(B1.model.Model, "cartoon_set")
    run_mode(B2.model.Model, "cartoon_set")

if __name__ == "__main__":
    main()
