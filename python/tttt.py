
import platform

system_name = platform.system()

if system_name == "Windows":
    print("windows")
elif system_name == "Linux":
    print("Linux")
else:
    print(system_name)

print("sad")

