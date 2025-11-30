from A4codes import learn, compute_accuracy

model = learn("/Users/eddieanokye/School/COMP 3105/COMP3105_A4/A4data/in-domain-train",
              "/Users/eddieanokye/School/COMP 3105/COMP3105_A4/A4data/out-domain-train")

acc_in = compute_accuracy("/Users/eddieanokye/School/COMP 3105/COMP3105_A4/A4data/in-domain-eval", model)
acc_out = compute_accuracy("/Users/eddieanokye/School/COMP 3105/COMP3105_A4/A4data/out-domain-eval", model)

print(f"In-domain accuracy: {acc_in:.4f}")
print(f"Out-domain accuracy: {acc_out:.4f}")