"""Native execution of retained tensor fields and temporal observations."""
from .execution_trace import current_binding, record_method


def _bound(source,field):
    field.check_state()
    if field.source is not None and field.source.source is not source:
        raise ValueError("tensor input requires its actual selected native source")
    binding = current_binding()
    if binding is not None:
        from .tensor_contracts import _bound as validate_reference
        validate_reference(field.source, binding)


def _execute(source,name,args):
    binding = current_binding()
    record_id = None if binding is None else binding.ref.record_id
    version = None if binding is None else binding.ref.record_version
    from rexgraph.tensor_field import TensorField,apply_tensor
    from rexgraph.tensor_moment import MomentSpan
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.temporal_field import TensorEvolution,moment_change
    from rexgraph.attachment_field import AttachmentField,common_attachment_observations
    from rexgraph.graded_metric import _fraction
    if name=="TENSOR_APPLY":
        action,field=args;_bound(source,field)
        result=apply_tensor(action,field)
    elif name=="NATIVE_RESPONSE":
        field,parameter,calculus=args;_bound(source,field)
        if field.grade is None:raise ValueError("native response requires a declared field grade")
        calculus=NativeFieldCalculus.from_rex(source) if calculus is None else calculus
        if calculus.source is not None and calculus.source is not source:
            raise ValueError("field calculus belongs to another selected source")
        result=apply_tensor(calculus.green(field.grade,_fraction(parameter)),field)
    elif name=="TENSOR_MOMENTS":
        kernel,left,right=args
        for side in (left,right):
            if side is None:
                continue
            fields=(side,) if isinstance(side,TensorField) else tuple(side)
            for field in fields:
                field.check_state()
                if field.source is None or field.source.source is source:
                    _bound(source,field)
                elif not any(field.source.matches(reference) and field.source.source is reference.source
                             for reference in kernel.endpoint_sources):
                    raise ValueError("moment field requires its declared endpoint source")
        if kernel.endpoint_sources and not any(reference.source is source for reference in kernel.endpoint_sources):
            raise ValueError("moment kernel must include the selected native source")
        result=kernel.evaluate(left,right)
    elif name=="FIELD_PAIR":
        left,right,pairing=args
        left.check_state();right.check_state()
        result=MomentSpan(left,right,pairing)
    elif name=="MOMENT_PAIR":
        moments,left,right=args;result=moments.pair(left,right)
    elif name=="MOMENT_SUPPORT":
        result=args[0].support()
    elif name=="MOMENT_CONTRACT":
        result=args[0].contracted_field()
    elif name=="TENSOR_SCALAR":
        field=args[0];field.check_state()
        if field.axes or len(field.space.keys)!=1:
            raise ValueError("scalar output requires an explicit contraction of all field axes")
        result=field.values[0]
    elif name=="TENSOR_DELTA":
        evolution,old,new=args;_bound(source,old)
        result=evolution.delta(old,new)
    elif name=="CHANNEL_FIELD":
        result=args[0].field(args[1]);result.check_state()
    elif name=="CHANNEL_TOTAL":
        result=args[0].total(args[1])
    elif name=="ATTACHMENTS":
        selected=AttachmentField.from_source(source,annotation_ids=args[0],roles=args[1],
                                            record_id=record_id, version=version)
        result=selected
    elif name=="SPAN_FIELD":
        selected,support,local,mode,amplitudes,refinement,scopes=args
        if selected.source is None or selected.source.source is not source:
            raise ValueError("span observation requires attachments from the bound native source")
        if binding is not None:
            from .tensor_contracts import _bound as validate_reference
            validate_reference(selected.source, binding)
        result=selected.observe(support=support,local=local,mode=mode,refinement=refinement,scopes=scopes).evaluate(amplitudes)
    elif name=="SPAN_DELTA":
        old,new,old_values,new_values,support,local,mode=args
        if old.source is None or old.source.source is not source:
            raise ValueError("span delta must bind its actual old selected source")
        if binding is not None:
            from .tensor_contracts import _bound as validate_reference
            validate_reference(old.source, binding)
        a,b=common_attachment_observations(old,new,support=support,local=local,mode=mode)
        old_values=old.amplitudes() if old_values is None else old_values
        new_values=new.amplitudes() if new_values is None else new_values
        if not isinstance(old_values,TensorField):old_values=old.amplitudes(old_values)
        if not isinstance(new_values,TensorField):new_values=new.amplitudes(new_values)
        result=TensorEvolution.from_observations(a,b).delta(old_values,new_values)
    elif name=="SECTOR_FIELDS":
        transport,old,new=args;_bound(source,old)
        result=transport.transport(old) if new is None else transport.reconstruct(old,new)
    elif name=="MOMENT_CHANGE":
        result=moment_change(*args)
    else:
        raise ValueError("unknown native tensor operation")
    observed={"method":"rational-retained-tensor-action","arithmetic":"rational","coefficient_domain":"Q"}
    digest=getattr(result,"coefficient_digest",None) or getattr(result,"declaration_digest",None)
    if digest is not None:observed["declaration_digest"]=digest
    source=getattr(result,"source",None)
    if source is not None:observed["field_source"]=source.as_record()
    endpoints=getattr(result,"endpoint_sources",())
    if endpoints:observed["endpoint_sources"]=tuple(s.as_record() for s in endpoints)
    if hasattr(result,"axes"):
        observed["retained_axes"]=tuple((a.name,a.keys) for a in result.axes)
    record_method(observed.pop("method"), **observed)
    return result


def tensor_apply(source, action, field):
    """Apply a native action while retaining tensor axes."""
    return _execute(source, "TENSOR_APPLY", (action, field))


def native_response(source, field, parameter=1, calculus=None):
    """Resolve the exact native field on its declared grade."""
    return _execute(source, "NATIVE_RESPONSE", (field, parameter, calculus))


def tensor_moments(source, kernel, left, right=None):
    """Retain channel fields and their selected moment forms."""
    return _execute(source, "TENSOR_MOMENTS", (kernel, left, right))


def field_pair(source, left, right, pairing):
    """Retain an exact cross moment with its support."""
    return _execute(source, "FIELD_PAIR", (left, right, pairing))


def moment_pair(source, moments, left, right):
    """Select one ordered channel pair without contracting its support."""
    return _execute(source, "MOMENT_PAIR", (moments, left, right))


def moment_support(source, moment):
    """Evaluate the retained support contributions of a moment."""
    return _execute(source, "MOMENT_SUPPORT", (moment,))


def moment_contract(source, moment):
    """Contract support while retaining the field axes."""
    return _execute(source, "MOMENT_CONTRACT", (moment,))


def tensor_scalar(source, field):
    """Read one explicitly contracted scalar coefficient."""
    return _execute(source, "TENSOR_SCALAR", (field,))


def tensor_delta(source, evolution, old, new):
    """Evaluate the declared temporal field channels."""
    return _execute(source, "TENSOR_DELTA", (evolution, old, new))


def channel_field(source, channels, name):
    """Select a named temporal field channel."""
    return _execute(source, "CHANNEL_FIELD", (channels, name))


def channel_total(source, channels, weights=None):
    """Apply explicit weights to compatible temporal channels."""
    return _execute(source, "CHANNEL_TOTAL", (channels, weights))


def attachments(source, annotation_ids=None, roles=None):
    """Select native attachments from the current source state."""
    return _execute(source, "ATTACHMENTS", (annotation_ids, roles))


def span_field(source, attachments, support="time", local=True, mode="sum", amplitudes=None, refinement=None, scopes=None):
    """Observe native attachment support without discarding its identity."""
    return _execute(source, "SPAN_FIELD", (attachments, support, local, mode, amplitudes, refinement, scopes))


def span_delta(source, old, new, old_values=None, new_values=None, support="time", local=True, mode="sum"):
    """Compare attachment fields on an explicit common support."""
    return _execute(source, "SPAN_DELTA", (old, new, old_values, new_values, support, local, mode))


def sector_fields(source, transport, old, new=None):
    """Retain exact sector transport and optional innovations."""
    return _execute(source, "SECTOR_FIELDS", (transport, old, new))


def moment_change(source, old_left, new_left, left_map, old_right, new_right, right_map, old_form, new_form):
    """Retain the exact finite change of a cross moment."""
    return _execute(source, "MOMENT_CHANGE", (old_left, new_left, left_map, old_right, new_right, right_map, old_form, new_form))


def install(register):
    register("TENSOR_APPLY")(tensor_apply)
    register("NATIVE_RESPONSE")(native_response)
    register("TENSOR_MOMENTS")(tensor_moments)
    register("FIELD_PAIR")(field_pair)
    register("MOMENT_PAIR")(moment_pair)
    register("MOMENT_SUPPORT")(moment_support)
    register("MOMENT_CONTRACT")(moment_contract)
    register("TENSOR_SCALAR")(tensor_scalar)
    register("TENSOR_DELTA")(tensor_delta)
    register("CHANNEL_FIELD")(channel_field)
    register("CHANNEL_TOTAL")(channel_total)
    register("ATTACHMENTS")(attachments)
    register("SPAN_FIELD")(span_field)
    register("SPAN_DELTA")(span_delta)
    register("SECTOR_FIELDS")(sector_fields)
    register("MOMENT_CHANGE")(moment_change)
