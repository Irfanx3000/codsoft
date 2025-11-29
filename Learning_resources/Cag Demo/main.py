from cag import cag_pipeline

print("Offline Cache-Augmented Generation")
print("----------------------------------")

while True:
    q = input("\nAsk: ")
    print("\n→", cag_pipeline(q))
