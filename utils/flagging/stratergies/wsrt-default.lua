--[[
 This is the WSRT strategy, version 2020-07-18.
 Author: André Offringa
]]--

function execute (input)

  --
  -- Generic settings
  --

  local base_threshold = 1.4  -- lower means more sensitive detection

  -- Options are: phase, amplitude, real, imaginary, complex
  local representation = "amplitude"
  local iteration_count = 3  -- how many iterations to perform?
  local threshold_factor_step = 2.0 -- How much to increase the sensitivity each iteration?
  
  --
  -- End of generic settings
  --

  local inpPolarizations = input:get_polarizations()
  
  input:clear_mask()
  
  -- For collecting statistics. Note that this is done after clear_mask(),
  -- so that the statistics ignore any flags in the input data. 
  local copy_of_input = input:copy()
  
  for ipol,polarization in ipairs(inpPolarizations) do
 
    local data = input:convert_to_polarization(polarization)

    data = data:convert_to_complex(representation)
    local original_data = data:copy()

    for i=1,iteration_count-1 do
      local threshold_factor = math.pow(threshold_factor_step, iteration_count-i)

      local sumthr_level = threshold_factor * base_threshold
      aoflagger.sumthreshold(data, sumthr_level, sumthr_level, true, true)

      -- Do timestep & channel flagging
      local chdata = data:copy()
      aoflagger.threshold_timestep_rms(data, 3.5)
      aoflagger.threshold_channel_rms(chdata, 3.0 * threshold_factor, true)
      data:join_mask(chdata)

      -- High pass filtering steps
      data:set_visibilities(original_data)
      local resized_data = aoflagger.downsample(data, 3, 3, false)
      aoflagger.low_pass_filter(resized_data, 21, 31, 1.6, 2.2)
      aoflagger.upsample(resized_data, data, 3, 3)

      -- In case this script is run from inside rfigui, calling
      -- the following visualize function will add the current result
      -- to the list of displayable visualizations.
      -- If the script is not running inside rfigui, the call is ignored.
      aoflagger.visualize(data, "Fit #"..i, i-1)

      local tmp = original_data - data
      tmp:set_mask(data)
      data = tmp

      aoflagger.visualize(data, "Residual #"..i, i+iteration_count)
      aoflagger.set_progress((ipol-1)*iteration_count+i, #inpPolarizations*iteration_count )
    end -- end of iterations

    aoflagger.sumthreshold(data, base_threshold, base_threshold, true, true)
 
    if input:is_complex() then
      data = data:convert_to_complex("complex")
    end
    input:set_polarization_data(polarization, data)

    aoflagger.visualize(data, "Residual #"..iteration_count, 2*iteration_count)
    
    aoflagger.set_progress(ipol, #inpPolarizations )
  end -- end of polarization iterations

  aoflagger.scale_invariant_rank_operator(input, 0.2, 0.2)
  aoflagger.threshold_timestep_rms(input, 4.0)
  input:flag_nans()
end

