use super::*;

pub fn compute_lookup_s_values<
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    delinearization_challenge: Fr,
) -> GpuResult<()> {
    assert!(S::PRODUCE_WITNESS);
    assert!(assembly.is_finalized);
    assert!(assembly.individual_table_canonical_sorted_entries.len() > 0);

    let (max_size_of_table, max_size_of_indexes, all_cols_size) = get_max_table_and_max_witness_size(assembly)?;

    let mut buffers = create_buffers_for_s_col_computation(manager, max_size_of_table, max_size_of_indexes, all_cols_size)?;

    let mut offset_in_result = MC::FULL_SLOT_SIZE - all_cols_size - 1;
    for (i, table_name) in assembly.known_table_names.iter().enumerate() {
        let ctx_id = i % MC::NUM_GPUS;

        get_s_part_from_table(
            manager,
            assembly,
            table_name,
            &mut buffers[ctx_id],
            delinearization_challenge,
            &mut offset_in_result,
            ctx_id
        )?;
    }

    async_free_buffers(manager, buffers)?;

    Ok(())
}

fn get_max_table_and_max_witness_size<
    S: SynthesisMode
>(
    assembly: &DefaultAssembly<S>,
) -> GpuResult<(usize, usize, usize)> {
    let mut max_size_of_indexes = 0;
    let mut max_size_of_table = 0;
    let mut total_size = 0;

    for table_name in assembly.known_table_names.iter(){
        let num_rows_of_original_table = assembly.
            individual_table_canonical_sorted_entries
            .get(table_name)
            .ok_or(GpuError::AssemblyError(format!("couldn't find the table {}", table_name)))?
            .len();

        let num_rows_of_witnesses = assembly
            .individual_table_entries
            .get(table_name)
            .ok_or(GpuError::AssemblyError(format!("couldn't file row indexes for the table {}", table_name)))?
            .len();

        if max_size_of_indexes < num_rows_of_witnesses + num_rows_of_original_table {
            max_size_of_indexes = num_rows_of_witnesses + num_rows_of_original_table;
        }

        if max_size_of_table < num_rows_of_original_table {
            max_size_of_table = num_rows_of_original_table;
        }

        total_size += num_rows_of_witnesses + num_rows_of_original_table;
    }

    Ok((max_size_of_table, max_size_of_indexes, total_size))
}

fn create_buffers_for_s_col_computation<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    table_length: usize,
    indexes_length: usize,
    all_cols_size: usize,
) -> GpuResult<Vec<(DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<u32>)>> {
    manager.new_empty_slot(PolyId::S, PolyForm::Values);

    let s_start = MC::FULL_SLOT_SIZE - all_cols_size - 1;
    let s_end = MC::FULL_SLOT_SIZE - 1;
    manager.set_values_with_range(
        PolyId::S,
        PolyForm::Values,
        Fr::zero(),
        0..s_start
    )?;
    manager.set_values_with_range(
        PolyId::S,
        PolyForm::Values,
        Fr::zero(),
        s_end..MC::FULL_SLOT_SIZE
    )?;

    let mut buffers = vec![];
    for ctx_id in 0..MC::NUM_GPUS {
        let table_buff = DeviceBuf::async_alloc_in_h2d(
            &manager.ctx[ctx_id],
            table_length,
        )?;
        let index_buff = DeviceBuf::async_alloc_in_h2d(
            &manager.ctx[ctx_id],
            indexes_length,
        )?;
        let sorted_idx_buff = DeviceBuf::async_alloc_in_exec(
            &manager.ctx[ctx_id],
            indexes_length,
        )?;

        buffers.push((table_buff, index_buff, sorted_idx_buff));
    }

    Ok(buffers)
}

fn get_s_part_from_table<
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    table_name: &str, 
    buffers: &mut (DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<u32>),
    delinearization_challenge: Fr,
    offset_in_result: &mut usize,
    ctx_id: usize
) -> GpuResult<()> {    
    let num_current_table_rows = assembly.individual_table_canonical_sorted_entries.get(table_name).unwrap().len();

    let indexes = assembly.individual_table_entries.get(table_name).unwrap();
    let indexes_len = indexes.len();

    let num_combined_rows = indexes_len + num_current_table_rows;

    upload_columns_lc(
        manager,
        assembly,
        table_name,
        &mut buffers.0,
        delinearization_challenge,
        ctx_id
    )?;

    upload_indexes(
        manager,
        assembly,
        table_name,
        &mut buffers.1,
        ctx_id
    )?;

    sort_indexes(
        manager,
        buffers,
        num_combined_rows,
        *offset_in_result,
        ctx_id
    )?;

    assign_columns(
        manager,
        buffers,
        num_combined_rows,
        *offset_in_result,
        ctx_id
    )?;

    *offset_in_result += num_combined_rows;

    Ok(())
}

fn upload_columns_lc<
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    table_name: &str, 
    table_buff: &mut DeviceBuf<Fr>,
    delinearization_challenge: Fr,
    ctx_id: usize
) -> GpuResult<()> {
    let original_table_values = assembly
        .individual_table_canonical_sorted_entries
        .get(table_name)
        .unwrap();

    let idx = manager.free_host_slot_idx().expect("no free host slot");

    let table = assembly.individual_table_canonical_sorted_entries.get(table_name).unwrap();
    let table_size = table.len();
    assert_eq!(table_size, original_table_values.len());
    assert!(table_size <= table_buff.len());
        
    let mut table_id_with_challenge = assembly.known_table_ids.get(table_name).unwrap().clone();
    table_id_with_challenge.mul_assign(&delinearization_challenge);

    let host_buff = &mut manager.host_slots[idx].0.get_values_mut()?[0..table_size];

    for (res, row) in host_buff.iter_mut().zip(original_table_values.iter()) {
        *res = table_id_with_challenge;
        res.add_assign(&row[2]);
        res.mul_assign(&delinearization_challenge);
        res.add_assign(&row[1]);
        res.mul_assign(&delinearization_challenge);
        res.add_assign(&row[0]);
    }

    unsafe {
        table_buff.async_copy_from_pointer_and_len(
            &mut manager.ctx[ctx_id],
            &host_buff[0] as *const Fr,
            0..table_size,
            table_size
        )?;
    }

    Ok(())
}

fn upload_indexes<
    S: SynthesisMode,
    MC: ManagerConfigs
>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    assembly: &DefaultAssembly<S>,
    table_name: &str,
    index_buff: &mut DeviceBuf<u32>,
    ctx_id: usize
) -> GpuResult<()> {
    let indexes = assembly.individual_table_entries.get(table_name).unwrap();
    let indexes_len = indexes.len();

    let idx = manager.free_host_slot_idx().expect("no free host slot");

    if indexes_len > 0 {
        unsafe {
            index_buff.async_copy_from_pointer_and_len(
                &mut manager.ctx[ctx_id],
                &indexes[0] as *const u32,
                0..indexes_len,
                indexes_len
            )?;
        }
    }
    let table = assembly.individual_table_canonical_sorted_entries.get(table_name).unwrap();
    let table_size = table.len();
    let start = MC::FULL_SLOT_SIZE - table_size;
    let host_buff = unsafe{ std::slice::from_raw_parts_mut(
        manager.host_slots[idx].0.as_ptr(start..MC::FULL_SLOT_SIZE) as *mut u32,
        table_size
    )};

    for (i, el) in host_buff.iter_mut().enumerate() {
        *el = i as u32;
    }

    unsafe {
        index_buff.async_copy_from_pointer_and_len(
            &mut manager.ctx[ctx_id],
            &host_buff[0] as *const u32,
            indexes_len..indexes_len+table_size,
            table_size
        )?;
    }

    manager.host_slots[ctx_id].0.write_event.record(
        manager.ctx[ctx_id].h2d_stream()
    )?;

    Ok(())
}

fn sort_indexes<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    buffers: &mut (DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<u32>),
    full_size: usize,
    offset_in_result: usize,
    ctx_id: usize
) -> GpuResult<()> {
    let mut stream = &mut manager.ctx[ctx_id].exec_stream;
    stream.wait(buffers.1.write_event())?;
    stream.wait(buffers.2.write_event())?;
    stream.wait(buffers.2.read_event())?;

    unsafe {
        let stream = manager.ctx[ctx_id].exec_stream.inner;
        let mem_pool = manager.ctx[ctx_id].mem_pool.expect("mem pool should be allocated");
        let values = buffers.1.as_mut_ptr(0..full_size) as *mut c_void;
        let sorted_values = buffers.2.as_mut_ptr(0..full_size) as *mut c_void;

        set_device(manager.ctx[ctx_id].device_id())?;
        let cfg = ff_sort_u32_configuration {
            mem_pool,
            stream,
            values,
            sorted_values,
            count: full_size as u32,
        };
        let result = ff_sort_u32(cfg);
        if result != 0 {
            panic!("sorted error {}", result);
        };
    }

    let mut stream = &mut manager.ctx[ctx_id].exec_stream;
    buffers.1.read_event.record(stream)?;
    buffers.2.write_event.record(stream)?;

    Ok(())
}

fn assign_columns<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    buffers: &mut (DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<u32>),
    full_size: usize,
    offset_in_result: usize,
    ctx_id: usize
) -> GpuResult<()> {
    let slot_idx = manager.get_slot_idx(PolyId::S, PolyForm::Values).unwrap();
    let device_id = manager.ctx[ctx_id].device_id();
    let stream = &mut manager.ctx[ctx_id].exec_stream;

    let ranges = get_ranges_for_assigments::<MC>(offset_in_result..offset_in_result+full_size);

    let mut offset = 0;
    for (idx, range) in ranges.into_iter().enumerate() {
        unsafe {
            let res_buff = &mut manager.slots[slot_idx].0[idx];

            stream.wait(buffers.0.write_event())?;
            stream.wait(res_buff.write_event())?;
            stream.wait(res_buff.read_event())?;
            stream.wait(buffers.2.write_event())?;
        
            let result = res_buff.as_mut_ptr(range.clone()) as *mut c_void;
            let variables = buffers.2.as_ptr(offset..offset);
            let assigments = buffers.0.as_ptr(0..0) as *const c_void;
        
            if range.len() > 0 {
                set_device(device_id)?;
                unsafe {
                    let result = ff_select(
                        assigments,
                        result,
                        variables,
                        range.len() as u32,
                        stream.inner,
                    );
                    if result != 0 {
                        return Err(GpuError::FFAssignErr(result));
                    };
                }

                res_buff.write_event.record(stream)?;
                buffers.2.read_event.record(stream)?;
                buffers.0.read_event.record(stream)?;
            }

            offset += range.len();
        }
    }

    Ok(())
}

fn async_free_buffers<MC: ManagerConfigs>(
    manager: &mut DeviceMemoryManager<Fr, MC>,
    buffers: Vec<(DeviceBuf<Fr>, DeviceBuf<u32>, DeviceBuf<u32>)>
) -> GpuResult<()> {
    for (ctx_id, buffers) in buffers.into_iter().enumerate() {
        let (mut table_buff, mut index_buff, mut sorted_idx_buff) = buffers;
        let stream = &mut manager.ctx[ctx_id].exec_stream;

        table_buff.async_free(stream)?;
        index_buff.async_free(stream)?;
        sorted_idx_buff.async_free(stream)?;
    }

    Ok(())
}

fn get_ranges_for_assigments<MC: ManagerConfigs>(range: Range<usize>) -> Vec<Range<usize>> {
    let mut res = vec![];

    for idx in 0..MC::NUM_GPUS {
        if range.start > (idx + 1) * MC::SLOT_SIZE || range.end < idx * MC::SLOT_SIZE {
            res.push(0..0);
        } else {
            let start = range.start.max(idx * MC::SLOT_SIZE) - idx * MC::SLOT_SIZE;
            let end = range.end.min((idx + 1) * MC::SLOT_SIZE) - idx * MC::SLOT_SIZE;
            res.push(start..end);
        }
    }

    res
}
