# initialize in serve mode - No calibration
# _get_quantized_vars - vars are quantized

call loop body 1 step and save the vars. 

That is Unet quantized vars.
apply this unet vars to unet model and the call serve on quantized. 